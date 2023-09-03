# -*- coding: utf-8 -*-
import glob
from os.path import join
from os import sep as osSep
from torchvision.transforms import ToTensor, Compose, ColorJitter
from torch.utils.data import Dataset
import torch
import numpy as np
from utils import PadImgToGivenSize, ImgProportionScale, TensorToImg
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle

# Dataloader for train, validation and test
class createDataLoader(object):
    def __init__(self, config):
        self.mode = config.MODE
        self.train_folder = config.TRAIN_FOLDER
        self.test_folder = config.TEST_FOLDER
        self.val_ratio = config.VAL_RATIO
        self.dbRoot = config.DB_ROOT
        self.dbName = config.DB_NAME
        self.img_size = config.IMG_SIZE
        self.batch = config.BATCH
        self.save_path = join(config.OUTPUT_DIR, "config")
    
    def build(self):
        print('Building {} dataloader...'.format(self.mode))
        img_list = []
        if(self.mode == "train"):
            for fold in self.train_folder:
                for img in glob.glob(join(self.dbRoot, self.dbName, fold, '*.jpg')):
                    img_list.append(img)        
            # random split to train and val set
            train_data, val_data = train_test_split(img_list, random_state=777, train_size=1-self.val_ratio)
            
            # save 
            # TODO : Recover train and val set by given file
            with open(join(self.save_path, "train_val_list.pkl"), "wb") as f:
                pickle.dump([train_data, val_data], f)
            
            train_db = ClassFromDataName(train_data, img_size=self.img_size)
            train_db.print_data_info()
            val_db = ClassFromDataName(val_data, img_size=self.img_size, mode='val')
            val_db.print_data_info()
            
            # create dataloader
            train_dataloader = DataLoader(
                train_db,
                batch_size=self.batch, #128
                shuffle=True,
                pin_memory=True)
            val_dataloader = DataLoader(
                val_db,
                batch_size=self.batch, #128
                shuffle=False,
                pin_memory=True)
            return train_dataloader, val_dataloader
        elif(self.mode == 'test'):
            for fold in self.test_folder:
                for img in glob.glob(join(self.dbRoot, self.dbName, fold, '*.jpg')):
                    img_list.append(img)
            test_db = ClassFromDataName(img_list, img_size=self.img_size, mode='test')
            test_db.print_data_info()
    
            test_dataloader = DataLoader(
                test_db,
                batch_size=self.batch, #128
                shuffle=False,
                pin_memory=True)
    
            return test_dataloader
class ClassFromDataName(Dataset):
    def __init__(self, data_list, split=".", img_size=512, colorJit = True, mode = 'train'):
        self.split = split
        self.img_size = img_size
        self.data_list = data_list
        self.mode = mode
        
        # create transform
        # TODO : Make data augmentations more flexible
        transformList = [ImgProportionScale(img_size), PadImgToGivenSize(img_size)] 
        if (colorJit): transformList.append(ColorJitter(0.4, 0.4, 0.4, 0.1))
        transformList.append(ToTensor())
        self.transform = Compose(transformList)
        
        # for debugging
        #self.showImg = TensorToImg(r'C:\Users\user\Desktop\Interview\aether\HW\temp')
    def load_data(self, img_path, load_label=True):
        img = self.transform(img_path)
        if(load_label):
            label = img_path.split(osSep)[-1].split('.')[0]
            if (label == 'cat'): label = torch.tensor(np.array([1,0]), dtype=torch.float32)
            else: label = torch.tensor(np.array([0,1]), dtype=torch.float32)
        else:
            label = -1
        # check image
        #img_name = img_path.split(osSep)[-1]
        #self.showImg.trans(img, img_name)
        
        return img, label
    def __len__(self):
        return len(self.data_list)    
    def __getitem__(self, index):
        img, label = self.load_data(self.data_list[index], load_label = (self.mode == "train" or self.mode == "val"))
        return {'img': img, 'label': label, 'indices': index, 'name': self.data_list[index]}#int(self.data_list[index].split(osSep)[-1].split('.')[0])}
    def print_data_info(self):
        print('\t{} set contains {} images.'.format(self.mode, len(self.data_list)))
    
    
    
        
            
        