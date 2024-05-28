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


def Get_Patient_Number(MainDir):
  index = MainDir.find("0")
  return MainDir[index:index+3]
  
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

def Padding_And_Resize_Data(Images_List, Type_List, TargetDimension):

    for i in range(len(Images_List)):
        # padding
        Images_List[i] = Array_ZeroPadding(Images_List[i], Images_List[i].shape[0:3] + (TargetDimension[-1],)) #(32,)

        # Resize
        if("label" in Type_List[i].lower()):
            Images_List[i] = IMP.Resize_5D(Images_List[i], TargetDimension,"Nearest")
        else:
            Images_List[i] = IMP.Resize_5D(Images_List[i], TargetDimension, "Bilinear")

    return Images_List

def Crop_Unannotated_Slices(Reference, Images_List):

    Temp_Reference = np.squeeze(Convert_Categorical_To_MergedLabels(Reference))
    MaskedData = MaskOut_ImageList(Temp_Reference, Images_List)
    return MaskedData

        
        
def Get_Batch_Atlasses(inLabelBatch, Atlas_MRI_List, Atlas_Label_List):

    Direction_Arr = Get_Batch_Direction(inLabelBatch)

    Left_Atlas_MRI, Right_Atlas_MRI = Atlas_MRI_List[0], Atlas_MRI_List[1]
    Left_Atlas_Label, Right_Atlas_Label = Atlas_Label_List[0], Atlas_Label_List[1]

    # repeat labels
    Repeated_Control_Label = Repeat_Atlas(Direction_Arr, Left_Atlas_Label, Right_Atlas_Label)

    # repeat images
    if(isinstance(Left_Atlas_MRI, list) ):
        pass # has to be defined
    else:
        Repeated_Control_Image = Repeat_Atlas(Direction_Arr, Left_Atlas_MRI, Right_Atlas_MRI)

        return Repeated_Control_Image[:], Repeated_Control_Label[:]



def Repeat_Atlas(Direction_Arr,Left_Atlas,Right_Atlas):

    Repeated_Control = np.repeat(Left_Atlas, len(Direction_Arr), axis=0)  # 8

    Right_Index = np.transpose(np.where(Direction_Arr == 1)) #"R"
    for i in Right_Index:
        Repeated_Control[i[0], :] = Right_Atlas[:]

    return Repeated_Control[:]

def Get_Batch_Direction(inLabelBatch):

    Direction_Arr = np.zeros((len(inLabelBatch),))
    NewBatch = np.zeros(inLabelBatch.shape[0:4])

    if(len(inLabelBatch.shape) == 5):
        Temp = np.zeros(inLabelBatch.shape)
        for L in range(inLabelBatch.shape[-1]):
            Temp[:, :, :, :, L] = inLabelBatch[:, :, :, :, L] * L

        NewBatch = np.sum(Temp,axis=-1)
    
    else:
        NewBatch[:] = inLabelBatch[:]

    for i in range(len(inLabelBatch)):
        Direction_Arr[i] = Get_Image_Direction(NewBatch[i,:])

    return Direction_Arr[:]

def Get_Image_Direction(inLabel):
    # extract points of muscle 8 and 9
    Points_8 = np.transpose(np.where(inLabel == 8))  # z,y,x
    Points_9 = np.transpose(np.where(inLabel == 9))  # z,y,x

    # muscle 8 is on the most right side of the image for the left leg, while muscle 9 is on the opposite side
    Center_x_8 = np.mean(Points_8[:, 2])
    Center_x_9 = np.mean(Points_9[:, 2])

    if (Center_x_8 > Center_x_9):
        return 0 #"L" #Left
    else:
        return 1 #"R" #Right


def Import_mha(Path):
    inputImage = SimpleITK.ReadImage(Path)
    numpyImage = SimpleITK.GetArrayFromImage(inputImage)  # out z,y,x
    numpyImage = np.transpose(numpyImage, (-1, 1, 0))

    return numpyImage


def Import_Control(Directory_Path, Selected_Image_Types_List, Normalize_Flag="Unity", Load_npy_Flag = False):
    # this function is the main interface between the user and loading the control data.
    # you have to write the name of the image type correctly: Fat, Fat_Aug, Water, ...


    Image_Dictionary = {}

    for Image_Type in Selected_Image_Types_List:
        if(Load_npy_Flag):
            Image_Dictionary[Image_Type] = np.load(os.path.join(Directory_Path , Image_Type+".npy"))
        else:
            Image_Dictionary[Image_Type] = Import_mha(os.path.join(Directory_Path , Image_Type+".mha"))
        Image_Dictionary[Image_Type] = Image_Dictionary[Image_Type].astype(float)

        if "_Aug" in Image_Type:
            if(Load_npy_Flag):
                Labels = np.load(os.path.join(Directory_Path , "Labels_Aug_13.npy"))
            else:
                Labels = Import_mha(os.path.join(Directory_Path , "Labels_Aug_13.mha"))
        else:
            if(Load_npy_Flag):
                Labels = np.load(os.path.join(Directory_Path, "Labels_13.npy"))
            else:
                Labels = Import_mha(os.path.join(Directory_Path, "Labels_13.mha"))

    # normalize the image dictionary
    Normalize_Dictionary(Image_Dictionary, NormType=Normalize_Flag)


    ## convert the labels atlas to one-hot vector
    if(len(Selected_Image_Types_List)> 1):
        Images = Concatenate_Images_And_Retrieve_AllData_In1Array([Image_Dictionary])
    else:
        Images = Image_Dictionary[Selected_Image_Types_List[0]]

    NewShape = (1,) + Labels.shape # add indexing
    Labels = np.reshape(Labels, NewShape)

    CategoricalLabels = Convert_4DLabels_Array(Labels)


    # padding
    if(len(Selected_Image_Types_List) == 1):
        NewShape = (1,)+Images.shape+(1,) # add indexing and channel
        Images = np.reshape(Images,NewShape)
    else:
        NewShape = (1,)+Images.shape # add indexing only
        Images = np.reshape(Images,NewShape)

    Images = Array_ZeroPadding(Images, [1,Images.shape[1], Images.shape[2], 32])#Dictionary_ZeroPadding(Image_Dictionary)
    CategoricalLabels = Array_ZeroPadding(CategoricalLabels, [1, Labels.shape[1], Labels.shape[2], 32])

    return Images, CategoricalLabels

def Correct_Order_of_OriginalAndAugmented_Data(inData):
    inShape = inData.shape
    Original_Index = [0,1,2,3,4,5,6,7,8,9,10,11,12,26,27,28,32,33,34,35,36]

    AllSet = list(range(inShape[0]))
    Augment_Index = [x for x in AllSet if x not in Original_Index]

    OutData = np.zeros(inShape)
    OutData[0:21,:] = inData[Original_Index,:]
    OutData[21::, :] = inData[Augment_Index, :]

    return OutData

def Import_Normalize_Padd_3D_Images(Input_Directory_List, Selected_Image_Types_List, Normalize_Flag="Unity"):
    # This function calls the following functions: upload, normalize, and padd for the data of each input directory
    # and returns a dictionary of the images and their corresponding labels

    # The inputs are:
    # Input_Directory_List: list of the directories where the input data are saved
    #                       like the dirctory of the training, testing and validation data
    # Selected_Image_Types_List: the list of image types that will be uploaded suh as
    #                            Fat, Water, IP, OP, and FatFraction

    Uploaded_Data_List = [] # list of dictionaries
    Uploaded_Labels_List = []
    for Dict_i,Directory_Path in enumerate(Input_Directory_List):
        # Load images and labels
        TempData, TempLabels = Iterate_OverImageType_AndUpload_3D_ImageAndLabel_from_NPY(Directory_Path, Selected_Image_Types_List)

        # normalize the image dictionary
        Normalize_Dictionary(TempData, NormType=Normalize_Flag)

        ## convert the labels atlas to one-hot vector
        #TempCategoricalLabels = Convert_Dictionary_Labels(TempLabels)
        TempCategoricalLabels = Convert_4DLabels_Array(TempLabels[0])

        # append the ut lists
        Uploaded_Data_List.append(TempData)
        Uploaded_Labels_List.append(TempCategoricalLabels)


    return Uploaded_Data_List, Uploaded_Labels_List


def Iterate_OverImageType_AndUpload_3D_ImageAndLabel_from_NPY(Directory_Path, Selected_Image_Types_List):
    # This function iterates over the Selected_Image_Types_List and uploads the corresponding npy files
    # in the given Directory_Path

    # The inputs are:
    # Directory_Path: strin. the path of the training, testing or validation data. This path includes subfolders
    #                 that have categories like Fat, water,... and the labels
    # Selected_Image_Types_List: the list of image types that will be uploaded suh as
    #                            Fat, Water, IP, OP, and FatFraction

    Images_Dictionary = {}
    Labels_Dictionary = []

    # 1- upload the images
    for Type_i,Type_Name in enumerate(Selected_Image_Types_List):

        Images_Dictionary[Type_Name] = Upload_3D_Image_from_NPY(Directory_Path, Type_Name)

    # 2- upload the labels
    head, tail = os.path.split(Directory_Path)
    #Labels_Dictionary[tail] = Upload_3D_Labels_from_NPY(Directory_Path)
    Labels_Dictionary.append(Upload_3D_Labels_from_NPY(Directory_Path))

    return Images_Dictionary, Labels_Dictionary


def Upload_3D_Image_from_NPY(Directory_Path, Selected_Image_Type):
    # This function uploads the corresponding npy files in the given Directory_Path

    name = Selected_Image_Type
    Images_Path = os.path.join(Directory_Path,"Images")

    # define the path to the original images
    Original_Images_Path = os.path.join(Images_Path, name)
    Original_Images_Path = os.path.join(Original_Images_Path, name + "_Images.npy")

    # define the path to the corresponding augmented versions
    Aug_Images_Path = os.path.join(Images_Path, "Aug_" + name)
    Aug_Images_Path = os.path.join(Aug_Images_Path, "Aug_" + name + "_Images.npy")

    # load the images
    temp_Data = np.load(Original_Images_Path)
    temp_Aug_Data = np.load(Aug_Images_Path)  # , allow_pickle = True

    # adjust the data type
    temp_Data = temp_Data.astype(float)
    temp_Aug_Data = temp_Aug_Data.astype(float)

    temp_Data = np.squeeze(temp_Data)
    temp_Aug_Data = np.squeeze(temp_Aug_Data)

    # concatenate the oriinal and augmented data
    temp_Concatenate = np.concatenate((temp_Data, temp_Aug_Data), axis=0)

    return temp_Concatenate #temp_Data


def Upload_3D_Labels_from_NPY(Directory_Path):
    # This function uploads the corresponding label npy files in the given Directory_Path


    # define the path to the original images
    Original_Labels_Path = os.path.join(Directory_Path, "Labels/Labels_Labels.npy")

    # define the path to the corresponding augmented versions
    Aug_Labels_Path = os.path.join(Directory_Path, "Labels_Aug/Labels_Aug_Labels.npy")


    # load the images
    temp_Data = np.load(Original_Labels_Path)
    temp_Aug_Data = np.load(Aug_Labels_Path)  # , allow_pickle = True

    # concatenate the oriinal and augmented data
    temp_Labels = np.concatenate((temp_Data, temp_Aug_Data), axis=0)

    return temp_Labels #temp_Data


# Image Normalization Functions
#-------------------------------
def Normalize_Dictionary(InDict, NormType="Unity"):
    Dict_Keys = InDict.keys()

    for Key in Dict_Keys: # Fat,Water, IP, OP
        TempImg = np.zeros((InDict[Key].shape[1::]))

        for Case in range(InDict[Key].shape[0]):
            TempImg[:] = InDict[Key][Case,:]

            if(NormType == "MeanSTD"):
                TempImg = NormalizeImage_MeanSTD(TempImg,str(Key))
            elif(NormType == "None"):
                pass
            elif(NormType == "Unity" or NormType != "MeanSTD"):
                TempImg = NormalizeImage_Unity(TempImg)

            InDict[Key][Case, :] = TempImg[:]

        del TempImg

    return

def NormalizeImage_Unity(InputImg):
    max_intensity = np.amax(InputImg)
    img = InputImg / float(max_intensity)
    return img[:]

def NormalizeImage_MeanSTD(InputImg,ImageCategory):
    img = (InputImg - Mean_SD_Dict[ImageCategory + "Mean"]) / float(Mean_SD_Dict[ImageCategory + "STD"])
    return img[:]

# Labels Functions
#-----------------
def Convert_Dictionary_Labels(InputDict):
    OutDict = {}
    Dict_Keys = InputDict.keys()

    for Key in Dict_Keys:  # Training, Testing, Validation

        OutDict[Key]  = Convert_4DLabels_Array(InputDict[Key])

    return OutDict


def Convert_4DLabels_Array(InputArr):
    No_Of_Labels = len(np.unique(InputArr))
    OutputArr = np.zeros((InputArr.shape+(No_Of_Labels,)))

    for Case in range(OutputArr.shape[0]):
        OutputArr[Case,:] = Convert_Label_To_One_Hot_Categories(InputArr[Case,:])

    return OutputArr[:]


def Convert_Label_To_One_Hot_Categories(InputLabel):
    Processed_Labels = K_Utils.to_categorical(np.squeeze(InputLabel))
    return Processed_Labels[:]


def Convert_Categorical_To_MergedLabels(Padded_Labels):
    NCL = Padded_Labels.shape[-1]
    Temp_Padded_Labels = np.zeros(Padded_Labels.shape)
    for L in range(NCL):
        Temp_Padded_Labels[:, :, :, :, L] = Padded_Labels[:, :, :, :, L] * L
    Merge_Padded_Labels = np.sum(Temp_Padded_Labels, axis=-1)

    return Merge_Padded_Labels[:]

# Merge Labels or Create less labels
#----------------------------------
def Merge_Labels(Nested_List):
    New_List = []
    for i in range(len(Nested_List)):
        New_List.append(Create_New_LessLabels(Nested_List[i]))

    return New_List

def Create_New_LessLabels(inputLabels):
    # this function Import_Controles all muscles in one label in order to create new training labels in order to test a hypothesis in Unet with attention layer
    MergedLabel = np.sum(inputLabels[:, :, :, :, 1:13], axis=-1)

    Tr_Shape = inputLabels.shape
    Less_Labels = np.zeros((Tr_Shape[0:4] + (3,)))

    Less_Labels[:, :, :, :, 0] = inputLabels[:, :, :, :, 0] # background
    Less_Labels[:, :, :, :, 1] = MergedLabel[:, :, :, :]    # muscles
    Less_Labels[:, :, :, :, 2] = inputLabels[:, :, :, :, -1] # connective tissue

    return Less_Labels


# Concatnate the input images
def Concatenate_Images_And_Retrieve_AllData_In1Array(List_Of_Dicts):
    # the input is list of dictionaries. each dictionary has images that will be concatenated in the last channel.
    # the output will be one array that includes all the input cases with concatenated images inn the last channel
    Concatenated_List = Concatenate_Images(List_Of_Dicts)
    BigArray = Convert_BigList_To_BigArray(Concatenated_List) #np.vstack(Concatenated_List)

    return BigArray[:]

def Convert_BigList_To_BigArray(InList):
    BigArr = InList[0]

    for i in range(1,len(InList)):
        BigArr = np.concatenate((BigArr,InList[i]), axis=0)#np.vstack((BigArr,InList[i]))

    return BigArr[:]

def Concatenate_Images(List_Of_Dicts):
    # this function concateinates the images per dictionaryand returnns a new list
    NewList = []

    for i in range(len(List_Of_Dicts)):
        Add_LastCahnnel_ToEachImage_InDict(List_Of_Dicts[i])
        NewList.append(Concatenate_Images_Per_Dict(List_Of_Dicts[i]))

    return NewList

def Add_LastCahnnel_ToEachImage_InDict(InDict):
    # this functin prepares the last channel

    for Key in InDict.keys():
        NewShape = InDict[Key].shape+(1,)
        InDict[Key] = np.reshape(InDict[Key],NewShape)


def Concatenate_Images_Per_Dict(InDict):
    Keys_List = list(InDict.keys())
    No_Of_ImageType = len(Keys_List)

    if(No_Of_ImageType > 1):
        TempArray = np.concatenate((InDict[Keys_List[0]], InDict[Keys_List[1]]), axis=-1)

        if(No_Of_ImageType > 2):
            for i in range(2,No_Of_ImageType):
                TempArray = np.concatenate((TempArray, InDict[Keys_List[i]]), axis=-1)

        return TempArray

    else:
        print("only 1 image type is available in the loaded image dictionary. Concatenate_Images_Per_Dict cannot concatenate it")
        return InDict[Keys_List[0]]

def Concatenate_Images_Per_List(InList):

    No_Of_ImageType = len(InList)

    if (No_Of_ImageType > 1):

        TempArray = np.concatenate((InList[0], InList[1]), axis=-1)

        if (No_Of_ImageType > 2):
            for i in range(2, No_Of_ImageType):
                TempArray = np.concatenate((TempArray, InList[i]), axis=-1)

            return TempArray

        else:
            print(
                "only 1 image type is available in the loaded image list. Concatenate_Images_Per_List cannot concatenate it")
            return


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
        print("The working directory is already ready")

    Path_Dir = {"path_to_Weights": path_to_Weights, "path_to_InitialWeights": path_to_InitialWeights,
                "path_to_Results": path_to_Results, "path_to_Refrences": path_to_Refrences,
                "path_to_Results_Train": path_to_Results_Train, "path_to_Results_Valid": path_to_Results_Valid,
                "path_to_Results_Test": path_to_Results_Test, "path_to_Refrences_Train": path_to_Refrences_Train,
                "path_to_Refrences_Valid": path_to_Refrences_Valid, "path_to_Refrences_Test": path_to_Refrences_Test}

    return Path_Dir


def Create_Tensorboard_Curves_Folder(TB_FolderPath):
    try:
        Model_1_Curves = TB_FolderPath #os.path.join(path_to_saved_model,"IPOPInputs_Batch4_Norm01_CombLoss")
        os.makedirs(Model_1_Curves)
    except:
        print("the folder: Model_1_Curves is already exist")

# Folds Functions
def Convert_5D_into_4D(inArr):

    Reshaped_Arr = np.rollaxis(inArr, 3, 1)
    Reshaped_Arr = np.reshape(Reshaped_Arr, [-1, inArr.shape[1], inArr.shape[2], inArr.shape[-1]])

    return Reshaped_Arr[:]

def Convert_4D_into_5D(inArr,No_Slices):
    No_of_Cases = inArr.shape[0]//No_Slices

    New_Arr = np.zeros((No_of_Cases, inArr.shape[1], inArr.shape[2], No_Slices, inArr.shape[-1]))
    print("No of cases will be ", No_of_Cases)
    print("Shape of NewArr will be ", New_Arr.shape)

    for i in range(No_of_Cases):
        Start_i = i*No_Slices
        End_i = Start_i + No_Slices

        New_Arr[i,:,:,:,:] = np.rollaxis(inArr[Start_i:End_i,:,:,:],0,3)


    return New_Arr

def Create_Fold_Train_Test_Validate_UnAugData(Data,Labels,Subset_Start_Index,Subset_End_Index, Convert5D_4D_Flag = False):


    Train_set,Test_Validate_Set =  Generate_Fold_Indexes(Data.shape[0], Subset_Start_Index, Subset_End_Index)

    # Train set
    Train_Data = Data[Train_set, :, :, :, :]

    Train_Labels = Labels[Train_set, :, :, :, :]

    # Validation set
    Validation_Data = Data[Test_Validate_Set, :, :, :, :]
    Validation_Labels = Labels[Test_Validate_Set, :, :, :, :]

    # Test set
    Test_Data = Data[Test_Validate_Set, :, :, :, :]
    Test_Labels = Labels[Test_Validate_Set, :, :, :, :]

    # # Undersample
    # Train_Data = block_reduce(Train_Data, block_size=(1, 2, 2, 1, 1), func=np.median)
    # Train_Labels = block_reduce(Train_Labels, block_size=(1, 2, 2, 1, 1), func=np.median)
    #
    # Validation_Data = block_reduce(Validation_Data, block_size=(1, 2, 2, 1, 1), func=np.median)
    # Validation_Labels = block_reduce(Validation_Labels, block_size=(1, 2, 2, 1, 1), func=np.median)
    #
    # Test_Data = block_reduce(Test_Data, block_size=(1, 2, 2, 1, 1), func=np.min)
    # Test_Labels = block_reduce(Test_Labels, block_size=(1, 2, 2, 1, 1), func=np.min)

    if(Convert5D_4D_Flag):
        Train_Data = Convert_5D_into_4D(Train_Data)
        Train_Labels = Convert_5D_into_4D(Train_Labels)

        Validation_Data = Convert_5D_into_4D(Validation_Data)
        Validation_Labels = Convert_5D_into_4D(Validation_Labels)

        Test_Data = Convert_5D_into_4D(Test_Data)
        Test_Labels = Convert_5D_into_4D(Test_Labels)

    Dict ={"Train_Data": Train_Data, "Train_Labels": Train_Labels,
           "Validation_Data": Validation_Data, "Validation_Labels": Validation_Labels,
           "Test_Data": Test_Data, "Test_Labels": Test_Labels}

    return Dict

def Create_Fold_Train_Test_Validate_Data(Data,Labels,Subset_Start_Index,Subset_End_Index, Convert5D_4D_Flag = False):

    Original_Data, Augmented_Data = Original_Augmented_Data_Decomposition(Data)
    Original_Labels, Augmented_Labels = Original_Augmented_Data_Decomposition(Labels)

    # Test_Validate_Set = list(range(Subset_Start_Index, Subset_End_Index))
    # AllSet = list(range(Original_Data.shape[0]))
    # Train_set = [x for x in AllSet if x not in Test_Validate_Set]
    Train_set,Test_Validate_Set =  Generate_Fold_Indexes(Original_Data.shape[0], Subset_Start_Index, Subset_End_Index)

    # Train set
    Temp_O = Original_Data[Train_set, :, :, :, :]
    Temp_A = Augmented_Data[Train_set, :, :, :, :]
    Train_Data = np.concatenate((Temp_O, Temp_A), axis=0)

    Temp_O = Original_Labels[Train_set, :, :, :, :]
    Temp_A = Augmented_Labels[Train_set, :, :, :, :]
    Train_Labels = np.concatenate((Temp_O, Temp_A), axis=0)

    # Validation set
    Validation_Data = Augmented_Data[Test_Validate_Set, :, :, :, :]
    Validation_Labels = Augmented_Labels[Test_Validate_Set, :, :, :, :]

    # Test set
    Test_Data = Original_Data[Test_Validate_Set, :, :, :, :]
    Test_Labels = Original_Labels[Test_Validate_Set, :, :, :, :]

    # # Undersample
    # Train_Data = block_reduce(Train_Data, block_size=(1, 2, 2, 1, 1), func=np.median)
    # Train_Labels = block_reduce(Train_Labels, block_size=(1, 2, 2, 1, 1), func=np.median)
    #
    # Validation_Data = block_reduce(Validation_Data, block_size=(1, 2, 2, 1, 1), func=np.median)
    # Validation_Labels = block_reduce(Validation_Labels, block_size=(1, 2, 2, 1, 1), func=np.median)
    #
    # Test_Data = block_reduce(Test_Data, block_size=(1, 2, 2, 1, 1), func=np.min)
    # Test_Labels = block_reduce(Test_Labels, block_size=(1, 2, 2, 1, 1), func=np.min)

    if(Convert5D_4D_Flag):
        Train_Data = Convert_5D_into_4D(Train_Data)
        Train_Labels = Convert_5D_into_4D(Train_Labels)

        Validation_Data = Convert_5D_into_4D(Validation_Data)
        Validation_Labels = Convert_5D_into_4D(Validation_Labels)

        Test_Data = Convert_5D_into_4D(Test_Data)
        Test_Labels = Convert_5D_into_4D(Test_Labels)

    Dict ={"Train_Data": Train_Data, "Train_Labels": Train_Labels,
           "Validation_Data": Validation_Data, "Validation_Labels": Validation_Labels,
           "Test_Data": Test_Data, "Test_Labels": Test_Labels}

    return Dict

def Create_Atlas(inLabels):
    if(len(inLabels.shape) == 5): # in case the input is one-hot encoded
        inLabels = Convert_Categorical_To_MergedLabels(inLabels)

    Atlas = np.zeros(inLabels.shape)


    if(inLabels.shape[0] > 20): # in case of including the augmented data
        Target_8 = [0,8,24,26]
        Target_20 = [12, 20, 27, 29] # the aug version of case 8
        Target_9 = [1,2,3,4,5,6,7,9,10,11,25,30,31,32,33,34]


        for Case in range(Atlas.shape[0]):
            if(Case in Target_8):
                Atlas[Case,:,:,:] = inLabels[8,:,:,:]
            elif(Case in Target_20):
                Atlas[Case, :, :, :] = inLabels[20, :, :, :]
            elif (Case in Target_9):
                Atlas[Case, :, :, :] = inLabels[9, :, :, :]
            else:
                Atlas[Case, :, :, :] = inLabels[21, :, :, :]

    else: # in case of excluding the augmented data
        Target_8 = [0, 8, 12, 14]

        for Case in range(Atlas.shape[0]):
            if (Case in Target_8):
                Atlas[Case, :, :, :] = inLabels[8, :, :, :]
            else:
                Atlas[Case, :, :, :] = inLabels[9, :, :, :]

    inShape = Atlas.shape
    Atlas = np.reshape(Atlas, [inShape[0], inShape[1], inShape[2], inShape[3], 1])

    return Atlas[:]

def Create_Fold_Train_Test_Validate_DataWithAtlas(Data,Labels,Subset_Start_Index,Subset_End_Index, Convert5D_4D_Flag = False):

    Original_Data, Augmented_Data = Original_Augmented_Data_Decomposition(Data)
    Original_Labels, Augmented_Labels = Original_Augmented_Data_Decomposition(Labels)

    Atlas = Create_Atlas(Labels)
    Atlas = Atlas/13.0
    Original_Atlas, Augmented_Atlas = Original_Augmented_Data_Decomposition(Atlas)

    # Test_Validate_Set = list(range(Subset_Start_Index, Subset_End_Index))
    # AllSet = list(range(Original_Data.shape[0]))
    # Train_set = [x for x in AllSet if x not in Test_Validate_Set]
    Train_set,Test_Validate_Set =  Generate_Fold_Indexes(Original_Data.shape[0], Subset_Start_Index, Subset_End_Index)

    # Train set
    Temp_O = Original_Data[Train_set, :, :, :, :]
    Temp_A = Augmented_Data[Train_set, :, :, :, :]
    Train_Data = np.concatenate((Temp_O, Temp_A), axis=0)

    Temp_O = Original_Labels[Train_set, :, :, :, :]
    Temp_A = Augmented_Labels[Train_set, :, :, :, :]
    Train_Labels = np.concatenate((Temp_O, Temp_A), axis=0)

    Temp_O = Original_Atlas[Train_set, :, :, :, :]
    Temp_A = Augmented_Atlas[Train_set, :, :, :, :]
    Train_Atlas = np.concatenate((Temp_O, Temp_A), axis=0)

    # Validation set
    Validation_Data = Augmented_Data[Test_Validate_Set, :, :, :, :]
    Validation_Labels = Augmented_Labels[Test_Validate_Set, :, :, :, :]
    Validation_Atlas = Augmented_Atlas[Test_Validate_Set, :, :, :, :]

    # Test set
    Test_Data = Original_Data[Test_Validate_Set, :, :, :, :]
    Test_Labels = Original_Labels[Test_Validate_Set, :, :, :, :]
    Test_Atlas = Original_Atlas[Test_Validate_Set, :, :, :, :]



    if(Convert5D_4D_Flag):
        Train_Data = Convert_5D_into_4D(Train_Data)
        Train_Labels = Convert_5D_into_4D(Train_Labels)

        Validation_Data = Convert_5D_into_4D(Validation_Data)
        Validation_Labels = Convert_5D_into_4D(Validation_Labels)

        Test_Data = Convert_5D_into_4D(Test_Data)
        Test_Labels = Convert_5D_into_4D(Test_Labels)

    Dict ={"Train_Data": Train_Data, "Train_Labels": Train_Labels,
           "Validation_Data": Validation_Data, "Validation_Labels": Validation_Labels,
           "Test_Data": Test_Data, "Test_Labels": Test_Labels,
           "Train_Atlas": Train_Atlas, "Test_Atlas": Test_Atlas, "Validation_Atlas": Validation_Atlas}

    return Dict


def Original_Augmented_Data_Decomposition(Data):
    inShape = list(Data.shape)
    inShape[0] = inShape[0]//2
    inShape = tuple(inShape)

    Original_Data = np.zeros(inShape)
    Augmented_Data = np.zeros(inShape)

    Original_Data[:,:,:,:,:] = Data[0:inShape[0],:,:,:,:]
    Augmented_Data[:, :, :, :, :] = Data[inShape[0]::, :, :, :, :]

    return Original_Data, Augmented_Data


def Generate_Fold_Indexes(FullDataLength, Fold_Start_Index, Fold_End_Index):

    Test_Validate_Set = list(range(Fold_Start_Index, Fold_End_Index))
    AllSet = list(range(FullDataLength))
    Train_set = [x for x in AllSet if x not in Test_Validate_Set]

    return Train_set, Test_Validate_Set

# Check Functions
#----------------
def saveEyeResult(save_path,npyfile):
    np.save(save_path , npyfile)

def Check_Loaded_Images(Data_List, ImageType = "Fat", Directory_Index = 0, Case_Index = 0, Slice_Index = 10, ShowFlag = True):
    # ImageType = ["Fat", "Water", "IP", "OP"]

    print("Size of returned Data_List= ", len(Data_List))

    print("Size of Dictionary No. {} in Data_List= ".format(str(Directory_Index)), len(Data_List[Directory_Index]))
    print("The keys in Dictionary No. {} are: ".format(str(Directory_Index)), Data_List[Directory_Index].keys())

    print("Size of the {} in the Dictionary No. {} in Data_List= ".format(ImageType,str(Directory_Index)), Data_List[Directory_Index][ImageType].shape)

    print("The maximum intensity in {} image of Directory {} is ".format(ImageType,str(Directory_Index)), np.amax(Data_List[Directory_Index][ImageType]))

    if(ShowFlag):
        TempImg = np.zeros((Data_List[Directory_Index][ImageType].shape[1:3]))
        TempImg[:] = np.squeeze(Data_List[Directory_Index][ImageType][Case_Index,:,:,Slice_Index])
        plt.imshow(TempImg)
        plt.show()

    return

def Check_Loaded_Labels(Labels_List, Directory_Index = 0, Case_Index = 0, Slice_Index = 10, Label_Index = 0,ShowFlag = True):
    # ImageType = ["Fat", "Water", "IP", "OP"]

    print("Size of returned Labels_List= ", len(Labels_List))

    print("Size of element No. {} in Labels_List= ".format(str(Directory_Index)), len(Labels_List[Directory_Index]))
    print("Shape of array in element No. {} in Labels_List= ".format(str(Directory_Index)), Labels_List[Directory_Index].shape)


    print("The maximum intensity in categorical label of element No. {} is ".format(str(Directory_Index)), np.amax(Labels_List[Directory_Index]))

    if(ShowFlag):
        TempImg = np.zeros((Labels_List[Directory_Index].shape[1:3]))
        TempImg[:] = np.squeeze(Labels_List[Directory_Index][Case_Index,:,:,Slice_Index,Label_Index])
        plt.imshow(TempImg)
        plt.show()


def Check_Show_Channel_Images(InImgArr, Case_Index, Slice_Index, No_of_Slices = 18):

    TempImg = np.zeros((InImgArr.shape[1:3]))

    if(len(InImgArr.shape) == 5): # case,x,y,z,c
        for c in range(InImgArr.shape[-1]):
            TempImg[:] = np.squeeze(InImgArr[Case_Index, :, :, Slice_Index, c])
            plt.imshow(TempImg)
            plt.show()

    elif(len(InImgArr.shape) == 4): # case*z,x,y,c
        Target_Index = Case_Index*No_of_Slices + Slice_Index
        for c in range(InImgArr.shape[-1]):
            TempImg[:] = np.squeeze(InImgArr[Target_Index, :, :, c])
            plt.imshow(TempImg)
            plt.show()

def Dictionary_ZeroPadding(inDict,TargetSahpe_ForSingleImage=[]):
    for K in inDict:
        inShape = TargetSahpe_ForSingleImage
        if(len(inShape) == 0):
            inShape = inDict[K].shape
        inDict[K] = MaskOut_ImageList(inDict[K], [inShape[0], inShape[1], inShape[2], 32])


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
