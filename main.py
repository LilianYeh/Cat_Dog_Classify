# -*- coding: utf-8 -*-
import sys
sys.path.append(r'C:\Users\user\Desktop\Interview\aether\HW')
from model.ResNet50 import ResNet50
from config import Config
from dataset.dataset import createDataLoader

if __name__ == '__main__':
    # load and display config
    # TODO : recover config setting by given file
    config = Config()
    config.save_config()
    config.show_config()
    
    # create dataloader
    db_creator = createDataLoader(config)
    if(config.MODE == "train"): train_loader, val_loader = db_creator.build()
    else: test_loader = db_creator.build()
    
    # build model
    model = ResNet50(config)
    model.build()
    
    # according to MODE perform train or test
    if(config.MODE == "train"):
        model.train(train_loader, val_loader)
    else:
        model.test(test_loader)
    
    #from torch.nn import CrossEntropyLoss
    #criterion = CrossEntropyLoss()
    #_ = model.val(val_loader, criterion, plot=True)