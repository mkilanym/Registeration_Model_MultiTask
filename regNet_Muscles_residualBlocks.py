import tensorflow as tf
from keras.models import Model
from keras.layers import UpSampling3D, concatenate, Cropping3D, ZeroPadding3D, MaxPooling3D
from keras.layers import Conv3D, LeakyReLU, Input, BatchNormalization, Lambda
from keras.initializers import RandomNormal
from keras.optimizers import *

from Dense3DSpatialTransformer_kilany import *
from keras.metrics import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping




def regNet(fixed_image, moving_image, moving_segmentation, kernel_size=3, num_reg_classes=3, addSMToReg=True, filters_multiplier=1.0):
    '''
    Description:
    create a multi-resolution V-net model with 3 layers. No. of kernels per layer is fixed, however can be increased with a fixed ration "filters_multiplier"
    the model's outputs are DVFs at each layer. 
    
    Inputs:
    fixed_image: tensor, shape=(batch,x,y,z,2). the MR-images of the target case
    moving_image: tensor, shape=(batch,x,y,z,2). the MR-images of the atlas
    moving_segmentation: tensor, shape=(batch,x,y,z,labels). the corresponding atlas segmentations
    
    kernel_size: int, 3D kernel size
    filters_multiplier: float, it is used to increae or decrease no. of kernels per layer with a fixed ratio
    
    Outputs:
    x_low_res_reg : list of tensors, len() = moving_segmentation.shape[-1]. list of low resolution DVFs
    x_mid_res_reg : list of tensors, len() = moving_segmentation.shape[-1]. list of medium resolution DVFs
    x_high_res_reg: list of tensors, len() = moving_segmentation.shape[-1]. list of high resolution DVFs 
    '''
    
    num_tasks= moving_segmentation.shape[-1]
    
    enc_nf = [32, 64, 128]  #[16,32,64] 
    enc_nf = [int(x*filters_multiplier) for x in enc_nf]


    fixed_image = _Conv3D_LReLU_BN(fixed_image, 3, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=True) 
    moving_image = _Conv3D_LReLU_BN(moving_image, 3, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=True) 
            
    x_in = concatenate([fixed_image, moving_image], axis=-1)
    if addSMToReg:
        x_in = concatenate([x_in, tf.to_float(moving_segmentation)], axis=-1)  # Third channel with moving segmentation.
        
    x0 = _Conv3D_LReLU_BN(x_in, enc_nf[0], kernel_size=kernel_size, strides=1, pad='same')  # (batch, (n)^3, 16)
    x1 = _Residual_Conv3D_BN_LReLU(x0, enc_nf[0], kernel_size=kernel_size, strides=1, pad='same')    # (batch, (n)^3, 16)
    x2 = MaxPooling3D(pool_size=(2, 2, 2))(x1)
    
    x3 = _Conv3D_LReLU_BN(x2, enc_nf[1], kernel_size=kernel_size, strides=1, pad='same')    # (batch, (n/2)^3, 32)
    x4 = _Residual_Conv3D_BN_LReLU(x3, enc_nf[1], kernel_size=kernel_size, strides=1, pad='same')    # (batch, (n/2)^3, 32)
    x5 = MaxPooling3D(pool_size=(2, 2, 2))(x4)
    
    x6 = [_Conv3D_LReLU_BN(x5, enc_nf[2], kernel_size=kernel_size, strides=1, pad='same') for i in range(num_tasks)]    # (batch, (n/4)^3, 64)
    x7 = [_Residual_Conv3D_BN_LReLU(x, enc_nf[2], kernel_size=kernel_size, strides=1, pad='same') for x in x6]    # (batch, (n/4)^3, 64)

    x9 = [UpSampling3D(size=(2, 2, 2))(x) for x in x7]  # (batch, (n/2)^3, 64)
    x9 = [concatenate([x, x4]) for x in x9]      # (batch, (n/2)^3, 96)
    x10 = [_Conv3D_LReLU_BN(x, enc_nf[1], kernel_size=kernel_size, strides=1, pad='same') for x in x9]   # (batch, (n/2)^3, 32)
    x11 = [_Residual_Conv3D_BN_LReLU(x, enc_nf[1], kernel_size=kernel_size, strides=1, pad='same') for x in x10]  # (batch, (n/2)^3, 32)

    x13 = [UpSampling3D(size=(2, 2, 2))(x) for x in x11]  # (batch, (n)^3, 32)
    x13 = [concatenate([x, x1]) for x in x13]      # (batch, (n)^3, 48)
    x14 = [_Conv3D_LReLU_BN(x, enc_nf[0], kernel_size=kernel_size, strides=1, pad='same') for x in x13]  # (batch, (n)^3, 16)
    x15 = [_Residual_Conv3D_BN_LReLU(x, enc_nf[0], kernel_size=kernel_size, strides=1, pad='same') for x in x14]  # (batch, (n)^3, 16)

    x_low_res_reg = [_Conv3D_LReLU_BN(x, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False) for x in x7] #(batch, (n/4-7)^3, num_classes)
    x_mid_res_reg = [_Conv3D_LReLU_BN(x, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False) for x in x11] # (batch, (n/2-18)^3, num_classes)
    x_high_res_reg = [_Conv3D_LReLU_BN(x, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False) for x in x15]  # (batch, (n-40)^3, num_classes)


    return x_high_res_reg, x_mid_res_reg, x_low_res_reg


def _Conv3D_LReLU_BN(x_in, nf, kernel_size=3, strides=1, pad='valid', lrelu=True, batch_norm=True):
    '''
    Description:
    perform convolution with or without batch normalization and activation layers 
    
    Inputs:
    x_in: 5D tensor
    nf: int, no. of filters
    kernel_size: int, the 3D kernel size    
    strides: int, specifying the stride length of the convolution
    pad: string, either "valid" or "same"
    lrelu: bool, apply LeakyReLU if True
    batch_norm: bool, apply batch normalization if True
    
    Outputs:
    x_out : 5D tensor
    '''
    
    init_var = RandomNormal(mean=0.0, stddev=0.02)
    x_out = Conv3D(nf, kernel_size=kernel_size, padding=pad, kernel_initializer=init_var, strides=strides)(x_in)
    
    if batch_norm:
        x_out = BatchNormalization(center=True, scale=True)(x_out)
    if lrelu:
        x_out = LeakyReLU(0.2)(x_out)
    return x_out

def Apply_Addition(var):
    return var[0]+var[1]

def _Residual_Conv3D_BN_LReLU(x_in, nf, kernel_size=3, strides=1, pad='valid', lrelu=True, batch_norm=True):
    '''
    Description:
    create a residual block with or without batch normalization and activation layers 
    
    Inputs:
    x_in: 5D tensor
    nf: int, no. of filters
    kernel_size: int, the 3D kernel size    
    strides: int, specifying the stride length of the convolution
    pad: string, either "valid" or "same"
    lrelu: bool, apply LeakyReLU if True
    batch_norm: bool, apply batch normalization if True
    
    Outputs:
    x_out : 5D tensor
    '''
    x1 = _Conv3D_LReLU_BN(x_in, nf, kernel_size, strides, pad, lrelu=True, batch_norm=True)
    x2 = _Conv3D_LReLU_BN(x1, nf, kernel_size, strides, pad, lrelu=False, batch_norm=True)

    Adding = Lambda(Apply_Addition)([x_in, x2])
    x_out = LeakyReLU(0.2)(Adding)
    return x_out