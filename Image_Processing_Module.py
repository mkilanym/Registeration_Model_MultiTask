import PIL
import numpy as np
import random as rn
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import map_coordinates, rotate
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter

from skimage.transform import PiecewiseAffineTransform, warp
from pylab import *
import SimpleITK as sitk
import itertools

def Array_ZeroPadding(Data,TargetSahpe_ForSingleImage):
    No_Channel = Data.shape[-1]
    NewShape = tuple(TargetSahpe_ForSingleImage)+(No_Channel,)

    Result = np.zeros(NewShape)
    for ch in range(No_Channel):
        Result[:,:,:,:,ch] = Image_ZeroPadding(Data[:,:,:,:,ch], TargetSahpe_ForSingleImage)

    return Result
    
    
def Image_ZeroPadding(Data,TargetSahpe):
    inShape = Data.shape
    Flag = 0
    if(len(inShape) == len(TargetSahpe)):
        for i in range(0,len(inShape)):
            if(inShape[i] <= TargetSahpe[i]):
                Flag += 1

        if(Flag == len(inShape)):
            Out = np.zeros((TargetSahpe))
            if(len(inShape) == 4):
                Out[(TargetSahpe[0]-inShape[0])//2:(TargetSahpe[0]+inShape[0])//2,(TargetSahpe[1]-inShape[1])//2:(TargetSahpe[1]+inShape[1])//2,(TargetSahpe[2]-inShape[2])//2:(TargetSahpe[2]+inShape[2])//2,(TargetSahpe[3]-inShape[3])//2:((TargetSahpe[3]+inShape[3])//2)] = Data
            elif (len(inShape) == 3):
                Out[(TargetSahpe[0] - inShape[0])/2: (TargetSahpe[0] + inShape[0])/2,
                (TargetSahpe[1] - inShape[1])/2: (TargetSahpe[1] + inShape[1])/2,
                (TargetSahpe[2] - inShape[2])/2: (TargetSahpe[2] + inShape[2])/2] = Data
            return Out
        else:
            return Data
    else:
        return Data
        
def Resize(inImageArray, Size,Interpolation):
    newSize = (inImageArray.shape[0],) + Size + (inImageArray.shape[-1],)
    outImageArray = np.zeros(newSize)
    for case in range(inImageArray.shape[0]):
        for z in range(inImageArray.shape[-1]):
            TempPIL = Convert_ndarray_to_PIL(inImageArray[case,:,:,z])
            if("Nearest" in Interpolation):
                outImageArray[case,:,:,z] = TempPIL.resize(Size,PIL.Image.NEAREST)
            else:
                outImageArray[case, :, :, z] = TempPIL.resize(Size, PIL.Image.BILINEAR)
    return outImageArray

def Resize_5D(inImageArray, Size,Interpolation):
    # size=(x,y,z)
    SizeXY = Size[0:2]
    newSize = (inImageArray.shape[0],) + Size + (inImageArray.shape[-1],)
    outImageArray = np.zeros(newSize)
    for case in range(inImageArray.shape[0]):
        for z in range(inImageArray.shape[-2]):
            for ch in range(inImageArray.shape[-1]):
                TempPIL = Convert_ndarray_to_PIL(inImageArray[case,:,:,z,ch])
                if("Nearest" in Interpolation):
                    outImageArray[case, :, :, z,ch] = TempPIL.resize(SizeXY,PIL.Image.NEAREST)
                else:
                    outImageArray[case, :, :, z,ch] = TempPIL.resize(SizeXY, PIL.Image.BILINEAR)
    return outImageArray
    
def elastic_transform_Intrinsic4DLoop(image_label_list, alpha=1.0, sigma=0.1):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    # assert len(image.shape) == 2

    #np.random.seed(5)

    image = image_label_list[0]
    Label = image_label_list[1]

    outImage = np.zeros(image.shape)
    outLabel = np.zeros(Label.shape)

    shape_3D = image.shape[1:4]
    shape_random = (20, 20, 20)
    print("shape_3D: ", shape_3D)


    indices = Create_Random_DVF(shape_random, shape_3D, sigma= sigma, alpha= alpha)

    # apply the elastic deformation
    for case in range(image.shape[0]):
        outImage[case, :] = Warp_Elastic_Deformation(image[case, :], indices, shape_3D, order=1)
        outLabel[case, :] = Warp_Elastic_Deformation(Label[case, :], indices, shape_3D, order=0)



    return outImage[:], outLabel[:]



def Warp_Elastic_Deformation(image, indices, shape_3D, order=0):
    outImage = np.zeros(image.shape)

    if (len(shape_3D) == 3):  # (x,y,z)
        for ch in range(image.shape[-1]):
            outImage[:, :, :, ch] = map_coordinates(image[:, :, :, ch], indices, order=order).reshape(shape_3D)

    elif((len(shape_3D) == 4 and shape_3D[0]==1)):
        for ch in range(image.shape[-1]):
            outImage[0, :, :, :, ch] = map_coordinates(image[0, :, :, :, ch], indices, order=order).reshape(shape_3D)

    else: #(1,x,y,z) or (case,x,y,z)
        for ch in range(image.shape[-1]):
            outImage[:, :, :, :, ch] = map_coordinates(image[:, :, :, :, ch], indices, order=order).reshape(shape_3D)

    return outImage[:]

def Create_Random_DVF(DVF_Grid_Shape, Target_Shape, sigma= 1, alpha= 1):

    shape_random = DVF_Grid_Shape
    shape_3D = Target_Shape

    # create DVF with shape = shape_random
    dx_20 = gaussian_filter((np.random.randint(10, size=shape_random)), sigma, mode="constant", cval=0) * alpha
    dy_20 = gaussian_filter((np.random.randint(10, size=shape_random)), sigma, mode="constant", cval=0) * alpha

    # create tuple of indices corresponding to the small DVF
    dx_index = np.linspace(0, shape_3D[0], shape_random[0])
    dy_index = np.linspace(0, shape_3D[1], shape_random[1])
    dz_index = np.linspace(0, shape_3D[2], shape_random[2])
    dx_dy_dz = (dx_index, dy_index, dz_index)
    # print("dx_20: ", dx_20.shape)

    # create indices for interpolation in order to expand dx_20 into dx with shape = shape3D
    src_cols = np.linspace(0, shape_3D[1], shape_3D[1])
    src_rows = np.linspace(0, shape_3D[0], shape_3D[0])
    src_z = np.linspace(0, shape_3D[2], shape_3D[2])

    x, y, z = np.meshgrid(src_rows, src_cols, src_z, indexing='ij')
    x_y_z = np.dstack([x.flat, y.flat, z.flat])[0]

    # interpolation
    dx = interpn(dx_dy_dz, dx_20, x_y_z)
    dy = interpn(dx_dy_dz, dy_20, x_y_z)
    # print("dx: ", dx.shape)

    indices = np.reshape(x.flat + dx, (-1, 1)), np.reshape(y.flat + dy, (-1, 1)), np.reshape(z.flat, (-1, 1))

    return indices





def Rotate_Batch(inBatch, Angle=30, order=0):
    outBatch = np.zeros(inBatch.shape)


    if(len(inBatch.shape) == 3): #x,y,z
        outBatch = sitk_rotation3d(inBatch, theta_z=Angle, order=order)
        # outBatch = Rotate_3DImage(inBatch,Angle)

    elif(len(inBatch.shape) == 4): #case,x,y,z
        for case in range(inBatch.shape[0]):
            outBatch[case,:] = sitk_rotation3d(inBatch[case,:,:,:], theta_z=Angle, order=order)
            # outBatch[case,:] = Rotate_3DImage(inBatch[case,:,:,:],Angle)

    elif(len(inBatch.shape) == 5): #case,x,y,z,channel
        for case in range(inBatch.shape[0]):
            for chanel in range (inBatch.shape[-1]):
                outBatch[case,:,:,:,chanel] = sitk_rotation3d(inBatch[case,:,:,:,chanel], theta_z=Angle, order=order)
                # outBatch[case,:,:,:,chanel] = Rotate_3DImage(inBatch[case,:,:,:,chanel])

    return outBatch[:]

def Rotate_3DImage(inImage3D,Angle=30):
    OutImage = np.zeros(inImage3D.shape)

    for z in range(inImage3D.shape[2]):
        TempPIL = Convert_ndarray_to_PIL(inImage3D[:,:,z])
        TempPIL = Rotate_Slice(TempPIL, Angle)
        OutImage[:,:,z] = np.array(TempPIL)

    return OutImage[:]


def Rotate_Slice(inPILIage,Angle=30):
    return inPILIage.rotate(Angle)

def Convert_ndarray_to_PIL(inImage):
    image_from_array = PIL.Image.fromarray(inImage)
    return image_from_array

def Scipy_Rotate(inBatch, Angle=0):
    outImage = np.zeros(inBatch.shape)

    for case in range(inBatch.shape[0]):
        outImage[case,:] = rotate(np.squeeze(inBatch[case,:]), Angle, mode="nearest", axes=(0,1), reshape=False)

    return outImage[:]
#-------------------------------------------------------------------------
# SimpleITK section
def matrix_from_axis_angle(a):
    """ Compute rotation matrix from axis-angle.
    This is called exponential map or Rodrigues' formula.
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(theta) +
    #      (1.0 - np.cos(theta)) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(theta))

    return R


def resample(image, transform, order):
    """
    This function resamples (updates) an image using a specified transform
    :param image: The sitk image we are trying to transform
    :param transform: An sitk transform (ex. resizing, rotation, etc.
    :return: The transformed sitk image
    """
    reference_image = image
    if(order==0):
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2)),
                                              int(np.ceil(depth/2))))


def sitk_rotation3d(inimage, theta_z=0, show=False, order = 0):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :param image: An sitk MRI image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param show: Boolean, whether or not the user wants to see the result of the rotation
    :return: The rotated image
    """
    image = sitk.GetImageFromArray(inimage,isVector=False)
    theta_z = np.deg2rad(theta_z)
    euler_transform = sitk.Euler3DTransform()
    # print(euler_transform.GetMatrix())

    image_center = get_center(image)
    euler_transform.SetCenter(image_center)

    direction = image.GetDirection()
    axis_angle = (direction[0], direction[3], direction[6], theta_z)
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix(np_rot_mat.flatten().tolist())
    resampled_image = resample(image, euler_transform, order)
    if show:
        slice_num = int(input("Enter the index of the slice you would like to see"))
        plt.imshow(sitk.GetArrayFromImage(resampled_image)[slice_num])
        plt.show()
    return sitk.GetArrayFromImage(resampled_image)




def sitk_DVF3d(inimage, alpha=1, sigma=0.1, random_state_int=None, order = 0):

    img = sitk.GetImageFromArray(inimage, isVector=False)

    # Generate random samples inside the image, we will obtain the intensity/color values at these points.
    num_samples = 400
    physical_points = []
    for pnt in zip(*[list(np.random.random(num_samples) * sz) for sz in img.GetSize()]):
        physical_points.append(img.TransformContinuousIndexToPhysicalPoint(pnt))
    # for pnt in zip(*[list(np.random.random(num_samples) * sz) for sz in img.GetSize()]):
    #     physical_points.append(img.TransformContinuousIndexToPhysicalPoint(pnt))


    # Create an image of size [num_samples,1...1], actual size is dependent on the image dimensionality. The pixel
    # type is irrelevant, as the image is just defining the interpolation grid (sitkUInt8 has minimal memory footprint).
    interp_grid_img = sitk.Image([num_samples] + [1] * (img.GetDimension() - 1), sitk.sitkUInt8)

    # Define the displacement field transformation, maps the points in the interp_grid_img to the points in the actual
    # image.
    displacement_img = sitk.Image([num_samples] + [1] * (img.GetDimension() - 1), sitk.sitkVectorFloat64,
                                  img.GetDimension())
    for i, pnt in enumerate(physical_points):
        displacement_img[[i] + [0] * (img.GetDimension() - 1)] = np.array(pnt) - np.array(
            interp_grid_img.TransformIndexToPhysicalPoint([i] + [0] * (img.GetDimension() - 1)))

    # Actually perform the resampling. The only relevant choice here is the interpolator. The default_output_pixel_value
    # is set to 0.0, but the resampling should never use it because we expect all points to be inside the image and this
    # value is only used if the point is outside the image extent.
    interpolator_enum = sitk.sitkNearestNeighbor #sitk.sitkLinear
    default_output_pixel_value = 0.0
    output_pixel_type = sitk.sitkFloat32 if img.GetNumberOfComponentsPerPixel() == 1 else sitk.sitkVectorFloat32
    resampled_image = sitk.Resample(img, interp_grid_img, sitk.DisplacementFieldTransform(displacement_img),
                                     interpolator_enum, default_output_pixel_value, output_pixel_type)
                                     
    return sitk.GetArrayFromImage(resampled_image)


def sitk_DVF3D_Simple(inimage, alpha=1, sigma=0.1, order = 0):

    src_cols = np.linspace(0, inimage.shape[1], 10)
    src_rows = np.linspace(0, inimage.shape[0], 10)
    src_z    = np.linspace(0, inimage.shape[2], 10)
    x, y, z= np.meshgrid(src_rows, src_cols, src_z, indexing='ij')
    # z = np.zeros(x.shape)
    # src = np.dstack([x.flat, y.flat, src_z.flat])[0]
    #
    #
    # dx = gaussian_filter((np.random.randint(40, size=x.shape)), sigma, mode="constant", cval=0) * alpha
    # dy = gaussian_filter((np.random.randint(40, size=x.shape)), sigma, mode="constant", cval=0) * alpha
    # nx = x + dx
    # ny = y + dy

    image = sitk.GetImageFromArray(inimage, isVector=False)
    image = sitk.Cast(image, sitk.sitkFloat64)


    # Random_arr = gaussian_filter((np.random.randint(40, size=(10,10,10,3))), sigma, mode="constant", cval=0) * alpha
    Random_arr = gaussian_filter((np.random.random((10, 10, 10, 3))), sigma, mode="nearest", cval=0) * alpha
    Random_arr[:,:,:,2] = 0 # no displacement in the z-direction

    Spacing = tuple(reversed(tuple([x / 10.0 for x in inimage.shape])))
    displacement_image = sitk.GetImageFromArray(Random_arr,isVector=True)
    displacement_image = sitk.Cast(displacement_image, sitk.sitkVectorFloat64)
    displacement_image.SetSpacing(Spacing)
    #displacement_image.SetOrigin(image.GetOrigin())


    DVF_transform = sitk.DisplacementFieldTransform(displacement_image)


    resampled_image = resample(image, DVF_transform, order)


    return sitk.GetArrayFromImage(resampled_image)


def value_func_3d(x, y, z):
    return 2 * x + 3 * y - z
def Test_Interpn():
    x = np.linspace(0, 4, 7)
    y = np.linspace(0, 5, 7)
    z = np.linspace(0, 6, 7)
    points = (x, y, z)
    # print(points)

    values = value_func_3d(*np.meshgrid(*points, indexing='ij'))


    point = np.array([[2.21, 3.12, 1.15],[2.21, 3.12, 1.15]])
    print(interpn(points, values, point))
    print(values.shape)

    # # points = np.array(np.dstack([x, y, z])[0])
    # # fn = RegularGridInterpolator(points, values)
    # # print(fn(points))
    #
    # x_m, y_m, z_m = np.meshgrid(*points, indexing='ij')
    # points_m = np.array(np.dstack([x_m.flat, y_m.flat, z_m.flat])[0])
    # values = []
    # for x in points_m:
    #     values.append(value_func_3d(*x))
    # print(len(values))
    # print(interpn(points_m, np.array(values), point))