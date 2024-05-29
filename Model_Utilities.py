from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import Model 
from data_2D import Import_Control
import numpy as np
import random as rn
import os
import tensorflow as tf

class Model_Class():
    def __init__(self, inModel): 
        '''
        Description: 
        initialize the class object, and create a model
        
        Inputs:
        inModel: keras model to be learned or used for prediction
                
        Parameters:
        
        '''       
        self.model = inModel
        
        
    def Determenistic(self, MySeed=42):
        '''
        Description: 
        seed all sources of randomness.
        
        Inputs:
        MySeed: int 
        '''
        np.random.seed(MySeed)
        rn.seed(MySeed)
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['PYTHONHASHSEED'] = str(0)
        tf.random.set_random_seed(MySeed)
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


    def Create_TB_File(self, TB_File_Path, Curves_Path):
        '''
        Description: 
        create a text file called "TB.txt" at the directory TB_File_Path. This file contains the command that is used to show the tensorboard curves
        
        Inputs:
        TB_File_Path: str, path where the text file will be created
        Curves_Path : str, path to the tensorboard results 
        '''
        
        FileName = os.path.join(TB_File_Path,"TB.txt")
        with open(FileName,"w") as file:
            file.write("export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; p3tf -m tensorboard.main --logdir '{}' --host res-hpc-gpu01 --port=8078".format(Curves_Path))

    
    def PrepareCallbacks(self, Paths_Dict, path_to_saved_model, SpecialString, EarlyStop_monitor="val_lambda_8_dice_loss"):
        '''
        Description:
        This function creates the callbacks that will be used to monitor the model performance.
        
        Inputs:
        #Paths_Dict         : dict of str, a dictionary includes all paths related to an experiment
        path_to_Weights    : str, the path to the saved model's weights 
        path_to_saved_model: str, it is the path to the tensorboard folder where the training and validation curves will be saved
        SpecialString      : str, a string to name the model's weights file
        EarlyStop_monitor  : str, the quantity to be monitored.
        
        Parameters:
        path_to_Weights: str, it is the path (including the file name and its extension) of the auto-saved weights
        
        Outputs:
        CallBacks          : list, list of callbacks
        '''        
        #1- create tensorboard folder
        TB_Path = os.path.join(path_to_saved_model,SpecialString)
        try:
            os.makedirs(TB_Path, exist_ok=True)
        except:
            print("The TB directory is already ready")
            
        self.Create_TB_File(Paths_Dict["path_to_Weights"], TB_Path)
        
        #2- define the path to the model's weights file 
        path_to_Weights = os.path.join(Paths_Dict["path_to_Weights"], 'CBModel1_{}.hdf5'.format(SpecialString))
        
        #3- create callbacks
        modelFWIPOP_checkpoint = ModelCheckpoint(path_to_Weights, monitor='val_loss', save_weights_only=False, period = 5, save_best_only=True)
        EarlyStop = EarlyStopping(monitor=EarlyStop_monitor, patience=100, mode="min", restore_best_weights=True) # monitor="val_loss" #monitor="val_dice_loss"
        tb = TensorBoard(log_dir=TB_Path, histogram_freq=10, write_graph=True, write_images=False)

        CallBacks = [modelFWIPOP_checkpoint, EarlyStop, tb]
        return CallBacks


    def Get_AtlasOrCase_Image_Label(self, Folder_Path, Selected_Image_Types_List):
        '''
        Description:
        This function loads the image and label data accroding to the source folder.
        If the folder is the preprocessed atlas, the content of the folder is usually more than 2 files, then load using Import_Control
        otherwise, the data is the output of a registration model, so load using numpy.
        
        Inputs:
        Folder_Path              : str, path of the case data
        Selected_Image_Types_List: list of str, list of the images' names to be loaded
        
        Outputs
        Image : 5-D numpy array, the MR-images in the format batch*x*y*z*channels. the channels = len(Selected_Image_Types_List) 
        Label : 5-D numpy array, the reference labels in the format batch*x*y*z*channels. the channels = 14 (12 muscles+ 1 background + 1 in-between fat)
        '''
        #1- read the contents in Folder_Path
        Contents_List = os.listdir(Folder_Path)

        if(len(Contents_List) == 2): # concatenated
            Label = np.load(os.path.join(Folder_Path, "Label_13.npy"))
            Image = np.load(os.path.join(Folder_Path, "MR_Images.npy"))
        else:
            Image, Label = Import_Control(Folder_Path, Selected_Image_Types_List, Normalize_Flag="MeanSTD", Load_npy_Flag=True)

        return Image[:], Label[:]


    def Prediction_Routine_Flow_From_Dict(self, Cases_Names_List, Paths_List, Selected_Image_Types_List, save_DVF_Flag, Train_Test_Set="Train"):
        '''
        Description:
        Perform prediction per case. the case name is passed to this function, loaded from the corresponding folders to perform the prediction.
        
        Inputs:
        Cases_Names_List         : list of str, names of the cases to be loaded
        Paths_List               : list of str, it includes: path to the atlas directory, cases directory, folder to save results, dictionary of paths related to the experiment
        Selected_Image_Types_List: list of str, list of the images' names to be loaded
        save_DVF_Flag            : bool, save DVF if True
        Train_Test_Set           : str, either "Train" or "Test"
        '''
        Atlas_Path = Paths_List[0] 
        Cases_Path = Paths_List[1] 
        ResultsFolder_Path_1 = Paths_List[2] 
        Model_1_Paths = Paths_List[3]
    
    
        for CaseName in Cases_Names_List:
            # load atlas and case
            Individual_Atlas_Path= os.path.join(Atlas_Path, CaseName)
            Individual_Case_Path = os.path.join(Cases_Path, CaseName)            
        
            Atlas_Image, Atlas_Label = self.Get_AtlasOrCase_Image_Label(Individual_Atlas_Path, Selected_Image_Types_List)
            Case_Image, Case_Label   = self.Get_AtlasOrCase_Image_Label(Individual_Case_Path, Selected_Image_Types_List)

            # predict
            results = self.model.predict([Case_Image, Atlas_Image, Atlas_Label])
            
            Results_Path = Model_1_Paths["path_to_Results_{}".format(Train_Test_Set)]
            Reference_Path = Model_1_Paths["path_to_Refrences_{}".format(Train_Test_Set)]            
            
            np.save(os.path.join(Results_Path, "{}_Fold_0_Conv_Label.npy".format(CaseName)), results[1])
            np.save(os.path.join(Reference_Path, "{}_Labels_Fold_0_Conv.npy".format(CaseName)), Case_Label)


            # save all inputs and outputs of the model per case in a separate folder
            subString = "Train" if "Train" in Train_Test_Set else "Test"
                  
            self.Save_Individual_Data(Model_1_Paths["path_to_Results"], ResultsFolder_Path_1, subString, CaseName, results, Case_Image, Atlas_Image, Atlas_Label)         

            if (save_DVF_Flag):
                print("Try to export DVF ---")
                self.Save_DVF(ResultsFolder_Path_1, subString, CaseName, Case_Image, Atlas_Image, Atlas_Label)
            
                

    def Save_Individual_Data(self, Individual_Folders_Path, ResultsFolder_Path, subString, CaseName, results, Case_Image, Atlas_Image, Atlas_Label):
        '''
        Description:
        Create a folder that contains a subfolder for each patient containing the corresponding results.
        This is the same hierarchy of the preprocessed-atlas folder.
        
        Inputs:
        Individual_Folders_Path  : str, path where the "Individual_Folders" folder will be created. This folder will include subfolders per case
        ResultsFolder_Path       : str, path where the "Input_Images", "_Control_Images" and "_Control_Labels" folders will be created. 
                                   These folders will contain numpy files corresponding to CaseName
        subString   : str, either "Train" or "Test" 
        CaseName    : str, name of a case
        results     : lit of numpy arrays, the model outputs. result[0] is list of warped mris, result[1] is 5-D 5-D numpy array shows the segmentations of the case.
        Case_Image  : 5-D numpy array, the MR-images of the target case in the format 1*x*.y*z*channels. the channels = len(Selected_Image_Types_List) 
        Atlas_Image : 5-D numpy array, the MR-images of the corresponding elastic-atlas in the format 1*x*.y*z*channels. the channels = len(Selected_Image_Types_List)
        Atlas_Label : 5-D numpy array, the labels of the corresponding elastic-atlas that will be deformed to match the target case.
                      Its format is 1*x*.y*z*channels. the channels = len(No. of labels)
        '''
        Organized_Results = os.path.join(Individual_Folders_Path, "Individual_Folders")
        Organized_Results = os.path.join(Organized_Results, CaseName)
        
        # save the model's outputs (warped moving labels)
        os.makedirs(Organized_Results, exist_ok=True)
        np.save(os.path.join(Organized_Results,"Label_13.npy") , results[1])
        np.save(os.path.join(Organized_Results, "MR_Images.npy"), results[0])
        
        # -----------------------------------------------------
        # save the MRIs of the target case (fixed images) 
        Images_Path = os.path.join(ResultsFolder_Path,"{}_Input_Images".format(subString))
        os.makedirs(Images_Path, exist_ok=True)
        np.save(os.path.join(Images_Path, "{}_Case_{}.npy".format(CaseName, subString)), Case_Image)

        # save the elastix-atlas-MRIs (moving images) that are corresponding to the case 
        Controls_Path = os.path.join(ResultsFolder_Path, "{}_Control_Images".format(subString))
        os.makedirs(Controls_Path, exist_ok=True)
        np.save(os.path.join(Controls_Path, "{}_Atlas_{}.npy".format(CaseName, subString)), Atlas_Image)

        # save the elastix-atlas-labels (moving labels)
        Controls_Path = os.path.join(ResultsFolder_Path, "{}_Control_Labels".format(subString))
        os.makedirs(Controls_Path, exist_ok=True)
        np.save(os.path.join(Controls_Path, "{}_Atlas_{}_Label.npy".format(CaseName, subString)), Atlas_Label)
        
            
    def Save_DVF(self, savePath, subString, CaseName, Case_Image, Atlas_Image, Atlas_Label):
        '''
        Description:
        save the full-size DVF.
        Note: this version of the function is optimized to work with the multi-task model. so it saves a DVF per muscle.
        
        Inputs:
        savePath    : str, path where a folder will be created inside to save DVFs
        subString   : str, either "Train" or "Test" 
        CaseName    : str, name of a case
        Case_Image  : 5-D numpy array, the MR-images of the target case in the format 1*x*.y*z*channels. the channels = len(Selected_Image_Types_List) 
        Atlas_Image : 5-D numpy array, the MR-images of the corresponding elastic-atlas in the format 1*x*.y*z*channels. the channels = len(Selected_Image_Types_List)
        Atlas_Label : 5-D numpy array, the labels of the corresponding elastic-atlas that will be deformed to match the target case.
                      Its format is 1*x*.y*z*channels. the channels = len(No. of labels)
        '''
        layer_names=[layer.name for layer in self.model.layers]
        
        TargetLayers = ["DVF_{}".format(i) for i in range(Atlas_Label.shape[-1])]
        outputs     = [self.model.get_layer(TargetLayer).output for TargetLayer in TargetLayers] 
        
        #TargetLayer = "conv3d_20"
        #if("conv3d_40" in layer_names):
        #    TargetLayer = "conv3d_40"
            
        DVFModel = Model(inputs= self.model.input, outputs=outputs)
        results = DVFModel.predict([Case_Image, Atlas_Image, Atlas_Label])

        print("len(DVF)= {} and DVF[0].shape= {}".format(len(results), results[0].shape))
         
        DVF_Path = os.path.join(savePath, "{}_DVF".format(subString))
        DVF_Path = os.path.join(DVF_Path, "{}".format(CaseName))
        os.makedirs(DVF_Path, exist_ok=True)
        for i in range(len(results)):
            np.save(os.path.join(DVF_Path, "{}_DVF.npy".format(i)), results[i])
        
        
def Iterative_Training_Routine(Model_Dict_Parameters, Paths_TBFolder_List, FOV_info_String, X, Y, No_SubEpochs = 5, Max_No_Cases_Per_Training = 2, No_Iterative_Training = 20, No_Epochs= 100):
    # X is [X, Repeated_Control_Image, Repeated_Control_Label]

    if(1):#("pretrained_weights" in Model_Dict_Parameters.keys()):
        print("saved model will be loaded")

        # 1- prepare call back list
        Paths_Dict = Paths_TBFolder_List[0]
        TBFolder_Path = Paths_TBFolder_List[1]
        Callback_List = PrepareCallbacks(Paths_Dict, TBFolder_Path, FOV_info_String)

        No_Data_Patches = np.round(len(Y)/Max_No_Cases_Per_Training)

        # 2- load the model
        TheModel = Registration_Model_ST(**Model_Dict_Parameters)

        for overAllEpoch in range(No_Epochs):
            print("overAllEpoch = {}".format(overAllEpoch))
            
            for iterativeTraining in range(No_Iterative_Training):
                for patch in range(0, len(Y), Max_No_Cases_Per_Training):

                    if((overAllEpoch + iterativeTraining + patch) > 0 ):
                        # 2- load the model
                        Weights_path = os.path.join(Paths_Dict["path_to_Weights"], 'Finalsaveweights' + FOV_info_String + '.hdf5')
                        Model_Dict_Parameters["pretrained_weights"] = Weights_path
                        TheModel = Registration_Model_ST(**Model_Dict_Parameters)


                    # 3- get the training data
                    start_p = patch
                    end_p = patch + Max_No_Cases_Per_Training
                    X_temp = [X[0][start_p:end_p,:], X[1][start_p:end_p,:], X[2][start_p:end_p,:]]
                    Y_temp = Y[start_p:end_p,:]

                    for i in range(3):
                      print("X_temp[0].shape= ", X_temp[i].shape)
                    print("Y_temp.shape= ", Y_temp.shape)

                    # 4- train
                    history = TheModel.fit(x=X_temp, y=Y_temp, epochs=No_SubEpochs)

                    # 5- save the modelś weights
                    # inModel.save(os.path.join(Model_Paths["path_to_Weights"], 'RegModel1_Finalsave_' + FOV_info_String))
                    TheModel.save_weights(os.path.join(Paths_Dict["path_to_Weights"],
                                                      'Finalsaveweights' + FOV_info_String + '.hdf5'))

                    # 6- clear memory
                    del history
                    
                    if(overAllEpoch < (No_Epochs-1) and iterativeTraining < (No_Iterative_Training-1) and patch < (len(Y)-Max_No_Cases_Per_Training)):
                        del TheModel
                        keras.clear_session()

        return TheModel

    else:
        print("no load")
def Training_Routine(inModel, Dict_Unet_Parameters, Data_Dict, Flag_DataGen, Paths_Dict, TBFolder_Path, FOV_info_String):

    #1- prepare Folds
    # Data_Dict = Create_Fold_Train_Test_Validate_Data(inData, inLabels, StartFold, EndFold,
    #                                             Convert5D_4D_Flag=Flag_2DUnet)
    #
    # # Train_Data = Data_Dict["Train_Data"]
    # # Train_Labels = Data_Dict["Train_Labels"]
    # #
    # # Validation_Data = Data_Dict["Validation_Data"]
    # # Validation_Labels = Data_Dict["Validation_Labels"]
    # #
    # # Test_Data = Data_Dict["Test_Data"]
    # # Test_Labels = Data_Dict["Test_Labels"]

    #2- prepare call back list
    Callback_List = PrepareCallbacks(Paths_Dict, TBFolder_Path, FOV_info_String)


    #3- train
    history = Perform_Training(inModel, Data_Dict, Dict_Unet_Parameters, Callback_List, Paths_Dict, FOV_info_String,
                     Flag_DataGen)

    return history



    
    
def Save_Results(inFoldDict,Model_Paths, FOV_info_String):

    Train_1_Results = inFoldDict["Train_Result"]
    Validate_1_results = inFoldDict["Validation_Result"]
    Test_1_results = inFoldDict["Test_Result"]


    saveEyeResult(os.path.join(Model_Paths["path_to_Results_Train"], "Model_" + FOV_info_String + "_TrainU.npy"), Train_1_Results)

    saveEyeResult(os.path.join(Model_Paths["path_to_Results_Valid"], "Model_" + FOV_info_String + "_ValidU.npy"), Validate_1_results)

    saveEyeResult(os.path.join(Model_Paths["path_to_Results_Test"], "Model_" + FOV_info_String + "_TestU.npy"),   Test_1_results)

    return 1


def Save_References(inFoldDict, Model_Paths, FOV_info_String = "", No_of_Slices = 23, Flag_Convert2D_into3D = False):

    Train_Labels = inFoldDict["Train_Labels"]
    Validation_Labels = inFoldDict["Validation_Labels"]
    Test_Labels = inFoldDict["Test_Labels"]

    if (Flag_Convert2D_into3D):
        Train_Labels = Convert_4D_into_5D(Train_Labels, No_of_Slices)  # [0:(5*No_of_Slices),:,:,:]
        Validation_Labels = Convert_4D_into_5D(Validation_Labels, No_of_Slices)
        Test_Labels = Convert_4D_into_5D(Test_Labels, No_of_Slices)

    saveEyeResult(os.path.join(Model_Paths["path_to_Refrences_Train"], "Train_Labels_" + FOV_info_String + ".npy"),
                  Train_Labels)
    saveEyeResult(os.path.join(Model_Paths["path_to_Refrences_Valid"], "Valid_Labels_" + FOV_info_String + ".npy"),
                  Validation_Labels)
    saveEyeResult(os.path.join(Model_Paths["path_to_Refrences_Test"], "Test_Labels_" + FOV_info_String + ".npy"),
                  Test_Labels)

    return 1


