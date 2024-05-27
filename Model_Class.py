import csv
from regNet_Muscles_residualBlocks import regNet 
from keras.layers import MaxPooling3D
from keras.models import Model 
from Custome_Metrics import *
from keras.backend.tensorflow_backend import set_session
from Dense3DSpatialTransformer_kilany import Warp

from keras.optimizers import *
from keras.metrics import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

class Model_Class():
    def __init__(self, modelConfigurationDict): 
        '''
        Description: 
        initialize the class object, and create a model
        
        Inputs:
        modelConfigurationDict: dictionary, includes all model parameters
                
        Parameters:
        self.modelConfigurationDict : dictionary, includes all model parameters
        self.model   : keras model
        '''       
        self.modelConfigurationDict = modelConfigurationDict
        self.model = self.__create_model(**self.modelConfigurationDict)
        
    def __create_model(pretrained_weights=None, kernel_size=3, Image_Channels=1, num_reg_classes=14, addSMToReg=True, LR=1e-4, bendingWeight=0.01, DiceWeight=0.95):
        '''
        Description: 
        create model. This function calls (regNet) from (regNet_Muscles_residualBlocks) where the model archetecture is saved, then create the model based on the retrieved DVFs from (regnet)
        
        Inputs:
        pretrained_weights: str, path to a saved model (*.hdf5)
        kernel_size       : int, 3D kernel size
        Image_Channels    : int, size of channels
        num_reg_classes   : int, no. of classes to be segmented
        addSMToReg        : bool, True if you want to cncatenate the moving segmentation with the input fixed and moving images in the first layer
        LR                : float, learning rate
        bendingWeight     : float, weight of the bending energy loss
        DiceWeight        : float, weight of the Dice loss
                
        Outputs:
        model   : keras model
        '''   
         
        session_conf  = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        #session_conf .gpu_options.allow_growth = True
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf ) #session = InteractiveSession(config=config)
        set_session(sess)

        # Define the inputs'shapes
        input_image_size = (None, None, None, Image_Channels)
        input_label_size = (None, None, None, num_reg_classes)

        # Create Input layers
        fixed_image = Input(input_image_size)
        moving_image = Input(input_image_size)
        moving_segmentation = Input(input_label_size)
    
        # Get multi-resolution DVFs   
        x_high_res_DVF, x_mid_res_DVF, x_low_res_DVF = regNet(fixed_image, moving_image, moving_segmentation, kernel_size, num_reg_classes=3, addSMToReg=addSMToReg)

        # Warp the segmentations
        x_low_res_Seg, x_mid_res_Seg, x_high_res_Seg = Perform_Warping(moving_segmentation, x_low_res_DVF, x_mid_res_DVF, x_high_res_DVF)
        
        # Warp the MRI
        x_low_res_mri, x_mid_res_mri, x_high_res_mri = Perform_Warping(moving_image, x_low_res_DVF, x_mid_res_DVF, x_high_res_DVF)

        # get the resolutions of fixed image
        mid_fixed_mri = MaxPooling3D(pool_size=(2, 2, 2))(fixed_image)
        low_fixed_mri = MaxPooling3D(pool_size=(2, 2, 2))(mid_fixed_mri)

        # create lists
        Moving_Segmentation_List = []
        Moving_Segmentation_List.append(x_low_res_Seg)
        Moving_Segmentation_List.append(x_mid_res_Seg)
        Moving_Segmentation_List.append(x_high_res_Seg)

        Moving_MRI_List = []
        Moving_MRI_List.append(x_low_res_mri)
        Moving_MRI_List.append(x_mid_res_mri)
        Moving_MRI_List.append(x_high_res_mri)
 
        Fixed_MRI_List = []
        Fixed_MRI_List.append(low_fixed_mri)
        Fixed_MRI_List.append(mid_fixed_mri)
        Fixed_MRI_List.append(fixed_image)

        
        model = Model(inputs=[fixed_image, moving_image, moving_segmentation], outputs=[x_high_res_mri, tf.stack(x_high_res_Seg, axis=-1)])

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        model.compile(optimizer=Adam(lr=LR), loss=[Resolution_NCC_Loss(Moving_MRI_List, Fixed_MRI_List, x_high_res_DVF, bendingWeight,DiceWeight), Resolution_Dice_NCC_Loss(Moving_Segmentation_List,Moving_MRI_List, Fixed_MRI_List, x_high_res_DVF, bendingWeight, DiceWeight, global_CC=False)], metrics ={"lambda_8":[dice_loss, Energy_Loss_List(x_high_res_DVF,"bending")]}) 
    
     
        model.summary()
        return model

        
    def Perform_Warping(inImage, inLowDVF, inMidDVF, inHighDVF):
        '''
        Description: 
        Apply DVFs to warp (inImage)
        
        Inputs:
        inImage   : 5D tensor, either MR-image or segmentations
        inLowDVF  : 4D tensor, DVF at low shape size
        inMidDVF  : 4D tensor, DVF at mid shape size
        inHighDVF : 4D tensor, DVF at high shape size

                
        Outputs:
        x_low_res_Seg  : list of tensors of warped image.
        x_mid_res_Seg  : list of tensors of warped image.
        x_high_res_Seg : list of tensors of warped image.
        '''  

        # Resize inImage to half and quarter of its x,y,z dimension
        mid_moving_segmentation = MaxPooling3D(pool_size=(2, 2, 2))(inImage)
        low_moving_segmentation = MaxPooling3D(pool_size=(2, 2, 2))(mid_moving_segmentation)
    
        x_low_res_Seg, x_mid_res_Seg, x_high_res_Seg = [], [], []
        
        # In case of segmentations: iterate over multi-DVF and segmentations'channels
        if(len(inLowDVF) == inImage.shape[-1]):
            # Apply each DVF to the corresponding segmentation                       
            for i in range(len(inLowDVF)):
                low_image = tf.expand_dims(low_moving_segmentation[...,i],-1)
                mid_image = tf.expand_dims(mid_moving_segmentation[...,i],-1)
                high_image = tf.expand_dims(inImage[...,i],-1)
                
                x_low_res_Seg.append(Lambda(Warp)([low_image, inLowDVF[i]]))
                x_mid_res_Seg.append(Lambda(Warp)([mid_image, inMidDVF[i]]))
                x_high_res_Seg.append(Lambda(Warp)([high_image, inHighDVF[i]]))
            
            ## Concatenate tensors on the last axis    
            #x_low_res_Seg  = tf.stack(x_low_res_Seg, axis=-1)
            #x_mid_res_Seg  = tf.stack(x_mid_res_Seg, axis=-1)
            #x_high_res_Seg = tf.stack(x_high_res_Seg, axis=-1)
                
        # In case of MR-images: iterate over multi-DVF       
        else:
            x_low_res_Seg.append(Lambda(Warp)([low_moving_segmentation, inLowDVF[i]]))
            x_mid_res_Seg.append(Lambda(Warp)([mid_moving_segmentation, inMidDVF[i]]))
            x_high_res_Seg.append(Lambda(Warp)([inImage, inHighDVF[i]]))
    
        return x_low_res_Seg, x_mid_res_Seg, x_high_res_Seg


    def Save_Weights(Model, FileName):
        with open(FileName, "a+") as f:
            writer = csv.writer(f)
            for layer in Model.layers:
                writer.writerow(layer.get_weights())

        print("Done saving the model weights")
    

    
'''    
def Normalized_Cross_Correlation_Loss(fixed,moving):
    meanFixed = tf.math.reduce_mean(fixed,axis=(0,1,2,3))
    meanMoving = tf.math.reduce_mean(moving, axis=(0, 1, 2, 3))
    stdFixed = tf.math.reduce_std(fixed, axis=(0, 1, 2, 3))
    stdMoving = tf.math.reduce_std(moving, axis=(0, 1, 2, 3))

    fixedSubtract = fixed - meanFixed
    movingSubtract = moving - meanMoving

    numerator = tf.math.reduce_mean(fixedSubtract * movingSubtract, (0,1,2,3) )

    loss = 1.0 - numerator/(stdFixed*stdMoving)

    return loss

def Mask_Fixed_MovingImages(fixed_swap, moving_swap, fixed_mask_swap):
    print("Mask_Fixed_MovingImages is applied")
    
    Thresholded_Mask = tf.cast(tf.math.greater(fixed_mask_swap, 0), tf.float32)  
    Mask = tf.math.reduce_sum(Thresholded_Mask[...,1:13], axis=-1)
    Ready_Mask = tf.cast(tf.math.greater(Mask, 0), tf.float32) 
    #Mask_expand = tf.expand_dims(Mask,-1)
    Mask_2 = tf.stack([Ready_Mask, Ready_Mask], axis=-1)
    
    Masked_Fixed = fixed_swap*Mask_2
    Masked_Moving = moving_swap*Mask_2
    
    return [Masked_Fixed, Masked_Moving]
    
def Resolution_Dice_Loss(Moving_Segmentation_List, DiceWeight=0.95):

    def loss(y_true, y_pred):
        mid_y_true = MaxPooling3D(pool_size=(2, 2, 2))(y_true)
        low_y_true = MaxPooling3D(pool_size=(2, 2, 2))(mid_y_true)

        Low_Dice_Loss = dice_loss(low_y_true, Moving_Segmentation_List[0])
        Mid_Dice_Loss = dice_loss(mid_y_true, Moving_Segmentation_List[1])
        High_Dice_Loss = dice_loss(y_true, y_pred)

        Total_Dice_Loss = DiceWeight*(Mid_Dice_Loss + High_Dice_Loss + Low_Dice_Loss)

        return Total_Dice_Loss

    return loss

'''

