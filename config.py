# -*- coding: utf-8 -*-
from os.path import join
from os import makedirs
from torch.cuda import is_available
import pickle
class Config(object):
    # experiment name, results are strored according to this
    EXP_NAME = "test_0902"
    # can input 'train' 'test'
    MODE = "test" 
    
    
    """ ---- dataset params ---- """
    # image size, all images in db will resize to IMG_SIZE x IMG_SIZE
    IMG_SIZE = 256
    # dataset root 
    DB_ROOT = r'C:\Users\user\Desktop\Interview\aether\HW\data'
    # dataset name
    DB_NAME = r'kaggle_Asirra'
    # train image folder(s)
    TRAIN_FOLDER = ['train']
    # test image folder(s)
    TEST_FOLDER = ['test1']
    # control how many images in train folder become validaion set , must in range (0.0, 1.0)
    VAL_RATIO = 0.2
    
    
    """ ---- model params ---- """
    # For resnet can accept "DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2", "FILE"
    #   *FILE : weight from the input WEIGHT_FILE 
    WEIGHT = "FILE"
    WEIGHT_FILE = r"C:\Users\user\Desktop\Interview\aether\HW\output\test_0902\model\epoch_8.pth.tar"#None
    # output root
    OUTPUT_ROOT = r'.\output'
    # number of class 
    CLASSES = 2
    
    """ ---- train params ---- """
    EPOCH = 10
    BATCH = 4
    LR = 3e-3
    # optimizer name, will catch the corresponding optimizer by name
    OPTIMIZER = "Adam"
    # schedular name, will catch the corresponding optimizer by name
    SCHEDULAR = "CyclicLR"
    CUDA = True if is_available() else False
    
    def __init__(self):
        # check config param 
        # TODO:  Completely parameter valid values check in config.py. Different models may have different valid ranges/values.
        assert self.WEIGHT in ["DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2", "FILE"], "Invalid weight setting"
        if(self.WEIGHT=="FILE" and not self.WEIGHT_FILE): assert "WEIGHT is set to FILE must provide WEIGHT_FILE"
        
        self.OUTPUT_DIR = join(self.OUTPUT_ROOT, self.EXP_NAME)
        if(self.MODE == "train"): 
            makedirs(self.OUTPUT_DIR,exist_ok=False)
            makedirs(join(self.OUTPUT_DIR, "model"),exist_ok=False)
            makedirs(join(self.OUTPUT_DIR, "config"),exist_ok=False)
            
    def save_config(self):
        with open(join(self.OUTPUT_DIR, "config", "config.pkl"), "wb") as f:
            pickle.dump(self, f)
    def show_config(self):
        print('------------ Config ------------')
        for item in dir(self):
            if(not item.startswith("_") and not callable(getattr(self, item))):
                print("\t{:15} {}".format(item, getattr(self, item)))
        print('------------ Config ------------')
        
    