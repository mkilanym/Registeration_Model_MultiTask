import tensorflow as tf

# NCC loss -------------------------------------------------------------------
def Resolution_NCC_Loss(Moving_MRI_List, Fixed_MRI_List, DVF, bendingWeight=0.01, DiceWeight=0.95):

    def loss(y_true, y_pred):
        Total_NCC_Loss = 0
        Total_Bending_Energy = 0
                
        Total_NCC_Loss = Resolution_LocalCC_Loss(Fixed_MRI_List, Moving_MRI_List)  
        
        for dvf in DVF:
            Total_Bending_Energy = Total_Bending_Energy + local_displacement_energy(dvf, "bending")     

        Total_Loss = (1.0-DiceWeight)*Total_NCC_Loss + bendingWeight*Total_Bending_Energy
        return 0.001*Total_Loss

    return loss

def Resolution_LocalCC_Loss(Fixed_MRI_List, Moving_MRI_List):
    Loss = 0
    
    for res in range(len(Fixed_MRI_List)):
    
        for i in range(len(Moving_MRI_List)):
        
            Loss = Loss + Local_Cross_Correlation_Loss(Fixed_MRI_List[res], Moving_MRI_List[res][i])
                     
    return Loss
        
def Local_Cross_Correlation_Loss(fixed, moving):
    #1- prepare the average filter
    Average_Kernel = tf.ones([3,3,3,1,1])*(1.0/729.0) # 9^3=729
    
    #2- prepare the input tensors by swapping depth and x-axis
    fixed_swap = tf.transpose(fixed, [0, 3, 1, 2, 4])  # swap axis depth and x --> [cases,z,x,y,ch]
    moving_swap = tf.transpose(moving, [0, 3, 1, 2, 4])  # swap axis depth and x --> [cases,z,x,y,ch]
    
    Local_Cross_Correlation = 0
    for i in range(2):
        #3- get the local mean of fixed and moving image 
        fixed_local_mean = tf.nn.conv3d(tf.expand_dims(fixed_swap[...,i],-1), Average_Kernel, strides=[1, 1, 1, 1, 1], padding='SAME')  
        moving_local_mean = tf.nn.conv3d(tf.expand_dims(moving_swap[...,i],-1), Average_Kernel, strides=[1, 1, 1, 1, 1], padding='SAME')

        #4- perform local subtractions
        fixed_subtract = tf.expand_dims(fixed_swap[...,0],-1) - fixed_local_mean
        moving_subtract = tf.expand_dims(moving_swap[...,0],-1) - moving_local_mean
    
        #5- get the numerator
        numerator =  tf.math.pow(tf.math.reduce_sum(fixed_subtract * moving_subtract),2)
    
        #6- get the denumerator
        denum_fixed = tf.math.reduce_sum(tf.math.pow(fixed_subtract,2))
        denum_moving = tf.math.reduce_sum(tf.math.pow(moving_subtract,2))
    
        Local_Cross_Correlation = Local_Cross_Correlation + -1*numerator/(denum_fixed * denum_moving+0.0001)
    
    return Local_Cross_Correlation
        
# --------------------------------------------------------------------------    
# Dice loss -------------------------------------------------------------------
def Resolution_Dice_NCC_Loss(Moving_Segmentation_List, Moving_MRI_List, Fixed_MRI_List, DVF, bendingWeight=0.01, DiceWeight=0.95, global_CC=True):

    def loss(y_true, y_pred):
        #1- calculate dice loss 
        mid_y_true = MaxPooling3D(pool_size=(2, 2, 2))(y_true)
        low_y_true = MaxPooling3D(pool_size=(2, 2, 2))(mid_y_true)

        Low_Dice_Loss  = dice_loss(low_y_true, tf.stack(Moving_Segmentation_List[0], axis=-1)) 
        Mid_Dice_Loss  = dice_loss(mid_y_true, tf.stack(Moving_Segmentation_List[1], axis=-1))
        High_Dice_Loss = dice_loss(y_true, tf.stack(Moving_Segmentation_List[2], axis=-1))

        Total_Dice_Loss = High_Dice_Loss + Mid_Dice_Loss + Low_Dice_Loss
        
        #2- claculate local CC loss
        Total_NCC_Loss = 0
        Total_NCC_Loss = Resolution_LocalCC_Loss(Fixed_MRI_List, Moving_MRI_List) #[low_y_true, mid_y_true, y_true]
        
        #3- claculate bending energy
        Total_Bending_Energy = 0
        #-------------------------------------------------------------------------------------
        # Note: compare the results of masked bending energy to unmasked bending energy   
        for i,dvf in enumerate(DVF):
            Threshold_Mask = tf.cast(tf.math.greater(Moving_Segmentation_List[2][i], 0), tf.float32)    
            Masked_DVF = Threshold_Mask*dvf
            
            Total_Bending_Energy = Total_Bending_Energy + local_displacement_energy(Masked_DVF, "bending")                   
        #-------------------------------------------------------------------------------------
        
        Total_Loss = DiceWeight*Total_Dice_Loss + (1.0-DiceWeight)*Total_NCC_Loss + bendingWeight*Total_Bending_Energy
        return Total_Loss

    return loss
        
def dice_loss(y_true, y_pred):

    Dice, variance = generalized_weighted_dice(y_true, y_pred)
    Ncl = tf.cast(y_pred.shape[-1], tf.int32) #keras.int_shape(y_pred)[4]
    #print("the shape of y_pred is: "+str(keras.int_shape(y_pred)))
    return Ncl-Dice #+ variance
    
def generalized_weighted_dice(y_true, y_pred):
    ## Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    ## Source of code: https://github.com/keras-team/keras/issues/9395
    keras_w = get_weights(y_true,y_pred)
    
    ## Compute gen dice coef:
    numerator = y_true * y_pred
    numerator = tf.math.reduce_sum(numerator, (0, 1, 2, 3)) #keras_w * keras.sum(numerator, (0, 1, 2, 3))
    print("The numerator = " + str(numerator))
    ## numerator = keras.sum(numerator)  # commented on 08/06/2020

    denominator = y_true + y_pred
    denominator = tf.math.reduce_sum(denominator, (0, 1, 2, 3)) #keras_w * keras.sum(denominator, (0, 1, 2, 3))
    print("The denominator = " + str(denominator))
    ## denominator = keras.sum(denominator)  # commented on 08/06/2020

    # gen_dice_coef = keras.sum(2 * numerator / denominator)
    gen_dice_coef = tf.math.reduce_sum(2 * numerator / denominator)

    gen_variance = tf.math.reduce_std(2 * keras_w * numerator / denominator)

    return  gen_dice_coef, gen_variance

def get_weights(y_true,y_pred):

    counts = tf.math.reduce_sum(y_true,axis=(0,1,2,3))
    Total  = tf.math.reduce_sum(counts)
    w = 1 - (counts / Total)
    print("The keras weights shape= ")
    print(str(w) )
    return w
# --------------------------------------------------------------------------
def Energy_Loss_List(DVF, energy_type="", energyWeight=1):
    Total_Loss = 0
    for dvf in DVF:
        Total_Loss =  Total_Loss + Energy_Loss(dvf, energy_type, energyWeight)
    return Total_Loss
    
def Energy_Loss(DVF, energy_type="", energyWeight=1):

    def loss(y_true, y_pred):

        #Mask = tf.math.reduce_sum(y_true[...,1:13],axis=-1)
        #Masked_DVF = tf.expand_dims(Mask,axis=4)*DVF
        Total_Loss =  energyWeight*local_displacement_energy(DVF, energy_type)

        return Total_Loss

    return loss


def local_displacement_energy(ddf, energy_type):

    def gradient_dx(fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(Txyz, fn):
        return tf.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], axis=4)

    def compute_gradient_norm(displacement, flag_l1=False):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        if flag_l1:
            norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return tf.math.reduce_mean(norms)

    def compute_Jacobian_matrix(displacement):
        dtdx = gradient_txyz(displacement, gradient_dx)
        dtdy = gradient_txyz(displacement, gradient_dy)
        dtdz = gradient_txyz(displacement, gradient_dz)

        df1dx = 1.0+dtdx[..., 0]
        df2dx = dtdx[..., 1]
        df3dx = dtdx[..., 2]

        df1dy = dtdy[..., 0]
        df2dy = 1.0+dtdy[..., 1]
        df3dy = dtdy[..., 2]

        df1dz = dtdz[..., 0]
        df2dz = dtdz[..., 1]
        df3dz = 1.0+dtdz[..., 2]

        Jacobian = df1dx * (df2dy * df3dz - df2dz * df3dy) - df1dy * (df2dx * df3dz - df2dz * df3dx) + df1dz * (
                    df2dx * df3dy - df2dy * df3dx)
        Jacobian = tf.round(Jacobian)

        return Jacobian

    def Get_Folds_Percentage(Jacobian):
        # mask the Jacobian
        Jacobian_Thr = tf.cast(tf.greater(Jacobian, 0), tf.keras.backend.floatx())
        Jacobian_Inv_Thr = 1.0-Jacobian_Thr

        return tf.math.reduce_mean(Jacobian_Inv_Thr)*100


    def compute_bending_energy(displacement):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        dTdxx = gradient_txyz(dTdx, gradient_dx)
        dTdyy = gradient_txyz(dTdy, gradient_dy)
        dTdzz = gradient_txyz(dTdz, gradient_dz)
        dTdxy = gradient_txyz(dTdx, gradient_dy)
        dTdyz = gradient_txyz(dTdy, gradient_dz)
        dTdxz = gradient_txyz(dTdx, gradient_dz)
        return tf.math.reduce_mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2) # try to use reduce_sum

    if energy_type == 'bending':
        energy = compute_bending_energy(ddf)
    elif energy_type == 'gradient-l2':
        energy = compute_gradient_norm(ddf)
    elif energy_type == 'gradient-l1':
        energy = compute_gradient_norm(ddf, flag_l1=True)
    elif energy_type == 'jacobian':
        Jacobian_Matrix = compute_Jacobian_matrix(ddf)
        energy = Get_Folds_Percentage(Jacobian_Matrix)
    else:
        raise Exception('Not recognised local regulariser!')

    return energy
