from keras import utils as K_Utils
from keras.preprocessing.image import ImageDataGenerator
import Image_Processing_Module as IMP
import numpy as np
import random as rn
import os
import matplotlib.pyplot as plt
import SimpleITK

Mean_SD_Dict ={"FatMean": 13.671, "FatSTD": 36.415,"WaterMean": 14.744, "WaterSTD": 38.247, "FatFractionMean": 0.003, "FatFractionSTD": 0.057, "IPMean": 27.962, "IPSTD": 58.128, "OPMean": 21.436, "OPSTD": 45.201}

def Get_CasePath_AtlasPath_CaseList(DataDirectory_Path, AtlasDirectory_Path):
    # set the name of folders where data will be loaded
    Cases_Path = os.path.join(DataDirectory_Path, "Cases")

    if (AtlasDirectory_Path == ""):  # the normal scinario where the atlas and cases are in the same main directory
        Atlas_Path = os.path.join(DataDirectory_Path, "Preprocessed_Atlas")
        Cases_Names_List = os.listdir(Cases_Path)

    else:  # the scinario of cascaded registration
        Atlas_Path = AtlasDirectory_Path
        Cases_Names_List = os.listdir(Atlas_Path)

    return Cases_Path, Atlas_Path, Cases_Names_List
    
    
# Folders management
def Create_Results_And_TB_Folders(ResultsPath, TBPath):
    Results_Path_Dir = Create_Working_Directory(ResultsPath)
    Create_Tensorboard_Curves_Folder(TBPath)
    return Results_Path_Dir


def Create_Working_Directory(MainFolder):
    path_to_Weights        = os.path.join(MainFolder, "Weights")
    path_to_InitialWeights = os.path.join(path_to_Weights, "Initials")
    path_to_Results        = os.path.join(MainFolder, "Results")
    path_to_Refrences      = os.path.join(MainFolder, "Refrences")

    path_to_Results_Train = os.path.join(path_to_Results, "Train")
    path_to_Results_Valid = os.path.join(path_to_Results, "Validate")
    path_to_Results_Test  = os.path.join(path_to_Results, "Test")

    path_to_Refrences_Train = os.path.join(path_to_Refrences, "Train")
    path_to_Refrences_Valid = os.path.join(path_to_Refrences, "Validate")
    path_to_Refrences_Test  = os.path.join(path_to_Refrences, "Test")

    try:
        os.makedirs(path_to_InitialWeights, exist_ok=True)

        os.makedirs(path_to_Results_Train,exist_ok=True)
        os.makedirs(path_to_Results_Valid, exist_ok=True)
        os.makedirs(path_to_Results_Test, exist_ok=True)

        os.makedirs(path_to_Refrences_Train, exist_ok=True)
        os.makedirs(path_to_Refrences_Valid, exist_ok=True)
        os.makedirs(path_to_Refrences_Test, exist_ok=True)

    except:
        print("The working directory already exists")

    Path_Dir = {"path_to_Weights": path_to_Weights, "path_to_InitialWeights": path_to_InitialWeights,
                "path_to_Results": path_to_Results, "path_to_Refrences": path_to_Refrences,
                "path_to_Results_Train": path_to_Results_Train, "path_to_Results_Valid": path_to_Results_Valid,
                "path_to_Results_Test": path_to_Results_Test, "path_to_Refrences_Train": path_to_Refrences_Train,
                "path_to_Refrences_Valid": path_to_Refrences_Valid, "path_to_Refrences_Test": path_to_Refrences_Test}

    return Path_Dir
    
    
def Create_Tensorboard_Curves_Folder(TB_FolderPath):
    Model_1_Curves = TB_FolderPath 
    os.makedirs(Model_1_Curves, exist_ok=True)

                 
def Convert_Categorical_To_MergedLabels(Padded_Labels):
    NCL = Padded_Labels.shape[-1]
    Temp_Padded_Labels = np.zeros(Padded_Labels.shape)
    for L in range(NCL):
        Temp_Padded_Labels[:, :, :, :, L] = Padded_Labels[:, :, :, :, L] * L
    Merge_Padded_Labels = np.sum(Temp_Padded_Labels, axis=-1)
    return Merge_Padded_Labels[:]
    
    
def MaskOut_ImageList(Reference, inList):
    outList = []
    for inImage in inList:
        outList.append(MaskOut_Image(Reference, inImage))
    return outList           
    
    
def MaskOut_Image(Reference, inImage):
    # 1- look for the labels
    xyz = np.where(Reference > 0)
    z = xyz[2]
    
    # 2- Create Mask
    Mask = np.zeros(inImage.shape)
    
    if(len(inImage.shape) == 4):        
        for zz in (z):
            Mask[:,:,:,zz] = 1
            
    else: # if xyz.shape[1] == 4
        for zz in (z):
            Mask[:,:,:,zz,:] = 1
        
    # 3- mask out the image
    return Mask*inImage                 