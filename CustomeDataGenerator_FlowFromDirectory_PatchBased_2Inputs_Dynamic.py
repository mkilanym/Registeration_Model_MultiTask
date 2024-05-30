from keras.utils import Sequence
from data_2D import Import_Control, Convert_Categorical_To_MergedLabels, MaskOut_ImageList
import numpy as np
import Image_Processing_Module as IMP
import os


class CustomeDataGenerator(Sequence):
    
    def __init__(self, Main_Data_Dir, Fold_Indexes, New_Atlas_Dir="", Patch_Size=(120, 120, 32), batch_size=1, dim=(240, 240, 32),
                 n_channels=4, n_classes=14, stride = 1, shuffle=True, AngleList=[], Selected_Image_Types_List=[],
                 Resize_Crop_Falg=False, Load_npy_Flag=False, Normalize_Flag="None"): #Patch_Size=(96, 96, 32), dim=(192, 192, 32)

        # Main_Data_Dir: the main folder that contains both of the patients and their corresponding preprocessed atlasses
        # Fold_Indexes : list of the cases that will be used to train a certain fold
        # Patch_Size   : the size of subimage that will be extracted from the loaded image
        # batch_size   : No. of subimages that will be concatenated to train in one go
        # dim          : the size of the loaded image
        # stride       : the amount of shift between 2 successive subimages. typically quarter of the loaded image dimension
        # AngleList    : list of angles for augmentation

        np.random.seed(42)
        # set directories
        self.Main_Data_Dir = Main_Data_Dir
        self.Cases_Path = os.path.join(Main_Data_Dir, "Cases")
        
        if(New_Atlas_Dir == ""):
            self.Atlases_Path = os.path.join(Main_Data_Dir, "Preprocessed_Atlas")
            self.Cascaded_Registration_Flag = False
        else:
            self.Atlases_Path = New_Atlas_Dir
            self.Cascaded_Registration_Flag = True
        self.Load_npy_Flag = Load_npy_Flag
        #self.Labels_path = Data_Path

        # set dimensions
        self.Fold_Indexes = Fold_Indexes
        self.dim = dim
        self.Patch_Size = Patch_Size
        self.batch_size = batch_size
        self.n_channels = len(Selected_Image_Types_List) #n_channels
        self.n_classes = n_classes
        self.stride = stride
        self.Selected_Image_Types_List = Selected_Image_Types_List
        self.Resize_Crop_Falg = Resize_Crop_Falg
        self.Normalize_Flag = Normalize_Flag

        # set augmentation parameters
        self.AngleList = AngleList
        self.shuffle = shuffle
        self.No_of_Input_MRI = len(self.Fold_Indexes)

        self.No_Patchs_Per_Axis, self.Total_No_Of_Patches_Per_Img = self.Get_Number_of_Patches_Per_Img()
        self.Indexes_list = np.arange(self.No_of_Input_MRI * self.Total_No_Of_Patches_Per_Img)

        # if self.shuffle == True:
        #     np.random.shuffle(self.Indexes_list)

        self.on_epoch_end()
        print("No of channels = " + str(self.n_classes))
        # print("Total No. of Patches = " , len(self.Patches_Indexes_list))



    def Get_Number_of_Patches_Per_Img(self):

        Last_Index = np.array(self.dim) - np.array(self.Patch_Size)

        self.X_Index_List = np.arange(0, Last_Index[0] + 1, self.stride) # the potential start x indecis per patch
        self.Y_Index_List = np.arange(0, Last_Index[1] + 1, self.stride) # the potential start y indecis per patch

        # make sure that the last Patch is included anyway
        self.X_Index_List = np.unique(np.append(self.X_Index_List, Last_Index[0]))
        self.Y_Index_List = np.unique(np.append(self.Y_Index_List, Last_Index[1]))
        
        No_Patchs_Per_Axis = (len(self.X_Index_List),len(self.Y_Index_List),1)
        Total_No_Of_Patches_Per_Img = int(np.prod(No_Patchs_Per_Axis))
        
        print("Last_Index = " , Last_Index)
        print("No_Patchs_Per_Axis = " , No_Patchs_Per_Axis)
        print("Total_No_Of_Patches_Per_Img = " , Total_No_Of_Patches_Per_Img)

        return No_Patchs_Per_Axis, Total_No_Of_Patches_Per_Img

    def Get_Patch_Index(self, Input_Index):
        return int(Input_Index/self.No_of_Input_MRI)

    def Get_Real_Img_Index(self,Input_Index):
        Patch_Index = self.Get_Patch_Index(Input_Index)
        Real_MRI_Index =  int(Input_Index - Patch_Index*self.No_of_Input_MRI)
        return Real_MRI_Index

    def Get_Start_And_End_X_Index_Per_Pach(self, Patch_Index):
        Start_X = 0
        End_X   = self.dim[0]
        if(self.No_Patchs_Per_Axis[0] > 1):
            Start_X = np.mod(Patch_Index,self.No_Patchs_Per_Axis[0])*self.Patch_Size[0]
            End_X   = Start_X + self.Patch_Size[0]

        return int(Start_X), int(End_X)

    def Get_Start_And_End_Y_Index_Per_Pach(self, Patch_Index):
        Start_Y = 0
        End_Y = self.dim[1]
        if (self.No_Patchs_Per_Axis[1] > 1):
            Start_Y = int(Patch_Index/ self.No_Patchs_Per_Axis[1]) * self.Patch_Size[1]
            End_Y = Start_Y + self.Patch_Size[1]

        return int(Start_Y), int(End_Y)



    def __len__(self):
        'Take all batches in each iteration'
        # return int(np.floor(self.Data_x.shape[0] / self.batch_size))  # int(len(self.file_list))
        return int(np.floor((self.No_of_Input_MRI * self.Total_No_Of_Patches_Per_Img) / self.batch_size))  # int(len(self.file_list))

    def __getitem__(self, index):
        'Get next batch'
        # Generate indexes of the batch
        sub_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # indexes = self.indexes[index:(index+1)]


        # single file
        file_list_temp = [self.Indexes_list[k] for k in sub_indexes]


        # Set of X_train and y_train
        X, y = self.__data_generation(file_list_temp)

        return X, y

    def on_epoch_end(self):
        # the method (on_epoch_end) is triggered once at the very beginning as well as at the end of each epoch
        self.indexes = np.arange(len(self.Indexes_list))

        if self.shuffle == True:
           np.random.shuffle(self.indexes)

        self.Type_of_Augmentation = np.random.randint(4)  # 0= no augmentation, 1=rotate only, 2=affine only, 3=rotate and affine
        if (self.AngleList):
            self.Angle = (np.random.uniform(-80,80,1)) # (np.random.uniform(-10,10,1)) #np.random.choice(self.AngleList)
        else:
            self.Angle = 0



    def Perform_Custome_Augmentation(self, Images_List, Labels_List):


        if ((self.Type_of_Augmentation == 1) or (self.Type_of_Augmentation == 3)):  # perform rotation only

            for i in range(len(Images_List)):
                Images_List[i] = IMP.Rotate_Batch(Images_List[i], Angle=self.Angle)
                Labels_List[i] = IMP.Rotate_Batch(Labels_List[i], Angle=self.Angle)


        if ((self.Type_of_Augmentation == 2) or (self.Type_of_Augmentation == 3)):  # perform affine only

            Tempx = np.concatenate(Images_List, axis=0)
            Tempy = np.concatenate(Labels_List, axis=0)

            Tempx, Tempy = IMP.elastic_transform_Intrinsic4DLoop([Tempx, Tempy],alpha=1, sigma=1) #IMP.Patch_Piecewise_Affine_Transform([Tempx, Tempy], alpha=1, sigma=1, order=1)

            for i in range(len(Images_List)):
                Images_List[i] = np.reshape(Tempx[i,], (1,) + Tempx.shape[1::])
                Labels_List[i] = np.reshape(Tempy[i,], (1,) + Tempy.shape[1::])


        return Images_List, Labels_List


    def Padding_And_Resize_Data(self, Images_List, Type_List):

        for i in range(len(Images_List)):
            # padding
            Images_List[i] = IMP.Array_ZeroPadding(Images_List[i], Images_List[i].shape[0:3] + (self.dim[-1],)) #(32,)

            # Resize
            if("label" in Type_List[i].lower()):
                Images_List[i] = IMP.Resize_5D(Images_List[i], self.dim,"Nearest")
            else:
                Images_List[i] = IMP.Resize_5D(Images_List[i], self.dim, "Bilinear")

        return Images_List

    def Crop_Unannotated_Slices(self, Reference, Images_List):

        Temp_Reference = np.squeeze(Convert_Categorical_To_MergedLabels(Reference))
        MaskedData = MaskOut_ImageList(Temp_Reference, Images_List)
        return MaskedData

    def __data_generation(self, image_list_temp):
        'Generates data containing batch_size samples'
        # Initialization
        Atlas_Image_Batch = np.empty((self.batch_size, *self.Patch_Size, self.n_channels))
        Atlas_Label_Batch = np.empty((self.batch_size, *self.Patch_Size, self.n_classes), dtype=int)

        Case_Image_Batch = np.empty((self.batch_size, *self.Patch_Size, self.n_channels))
        Case_Label_Batch = np.empty((self.batch_size, *self.Patch_Size, self.n_classes), dtype=int)


        # Generate data
        for i, ID in enumerate(image_list_temp):
            Case_X_Y_index = np.unravel_index(ID,
                                              (self.No_of_Input_MRI, len(self.X_Index_List), len(self.Y_Index_List)))


            Start_X = self.X_Index_List[Case_X_Y_index[1]]
            End_X = Start_X + self.Patch_Size[0]
            Start_Y = self.Y_Index_List[Case_X_Y_index[2]]
            End_Y = Start_Y + self.Patch_Size[1]
            Real_MRI_Index = Case_X_Y_index[0]

            # get the path of the atlas and corresponding case
            PatientNo = self.Fold_Indexes[Real_MRI_Index]
            AtlasDir = os.path.join(self.Atlases_Path, PatientNo)
            CaseDir  = os.path.join(self.Cases_Path, PatientNo)

            # load the atlas and case
            Case_Image,  Case_Label  = Import_Control(CaseDir,  self.Selected_Image_Types_List, Normalize_Flag=self.Normalize_Flag, Load_npy_Flag=self.Load_npy_Flag)
            
            if(self.Cascaded_Registration_Flag == False):
                Atlas_Image, Atlas_Label = Import_Control(AtlasDir, self.Selected_Image_Types_List, Normalize_Flag=self.Normalize_Flag, Load_npy_Flag=self.Load_npy_Flag)
            else:
                Atlas_Label = np.load(os.path.join(AtlasDir, "Label_13.npy"))
                Atlas_Image = np.load(os.path.join(AtlasDir, "MR_Images.npy"))
            
            # resize and crop data
            if(self.Resize_Crop_Falg):
                Images_List = self.Padding_And_Resize_Data([Atlas_Image, Atlas_Label, Case_Image,  Case_Label], ["image", "label", "image", "label"])
                MaskedData = self.Crop_Unannotated_Slices(Images_List[-1], Images_List[:-1])
                Atlas_Image, Atlas_Label, Case_Image, Case_Label = MaskedData[0], MaskedData[1], MaskedData[2], Images_List[-1]


            # in case of augmentation option
            if (self.Type_of_Augmentation > 0):
                Images_List, Labels_List = self.Perform_Custome_Augmentation([Atlas_Image, Case_Image],[Atlas_Label, Case_Label])

                Atlas_Image, Case_Image = Images_List[0], Images_List[1]
                Atlas_Label, Case_Label = Labels_List[0], Labels_List[1]


            # in case of no data augmentation or after applying the augmentation
            Atlas_Image_Batch[i,] = Atlas_Image[0, Start_X:End_X, Start_Y:End_Y, :, :]
            Atlas_Label_Batch[i,] = Atlas_Label[0, Start_X:End_X, Start_Y:End_Y, :, :]

            Case_Image_Batch[i,] = Case_Image[0, Start_X:End_X, Start_Y:End_Y, :, :]
            Case_Label_Batch[i,] = Case_Label[0, Start_X:End_X, Start_Y:End_Y, :, :]

        if(np.random.randint(2) == 1): # swap the case and atlas
            return [Atlas_Image_Batch, Case_Image_Batch, Case_Label_Batch], [Atlas_Image_Batch, Atlas_Label_Batch]
        else:
            return [Case_Image_Batch, Atlas_Image_Batch, Atlas_Label_Batch], [Case_Image_Batch, Case_Label_Batch]







