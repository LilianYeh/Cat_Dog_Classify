# Cats and Dogs Classification


## 1. Description
This project is to duild a deep-learning-based model to distinguish cats from dogs based on image data.
The dataset is publicly available on https://www.kaggle.com/competitions/dogs-vs-cats/data.

## 2. Requirements

## 3. Project overview
* data : Dataset folder. 
    ```bash
    ├── data
    │   ├── kaggle_Asirra
    │   │   ├── train
    │   │   └── test1
    │   ├── other dataset
    │   ...
    
    ```
* dataset : python file that define how to load data and create data loader
* model : python file that define model operation
* output : All results will be stored to this folder according to exp_name in config.py
    ```bash
    ├── output
    │   ├── exp_name1
    │   │   ├── config
    │   │   │   ├── config.pkl (experiment config setting)
    │   │   │   └── train_val_list.pkl (experiment train and validation image list)
    │   │   └── model
    │   │       ├── epoch_x.pth.tar
    │   │       ├── epoch_x_cm.png (confusion matrix image of epoch x)
    │   │       └── epoch_x_ROC.png (ROC curve image of epoch x)
    │   ├── exp_name2
    │   ...
    
    ```
* utils : some useful functions including image and result processing function.
* config.py : config setting. For detailed parameter description, please refer to the in-program comments.
* main.py : main file


## 4. Train
During training, the program will produce and save best model, confusion matrix, ROC curve with validation result on validation set.
1) Set the parameters in config file especially "EXP_NAME" and "MODE"
2) execute main.py by following command : 
```
    $ python main.py
```

## 5. Test
During testing, the program will produce a csv file that contains all testing images prediction results
1) Set the parameters in config file especially "EXP_NAME", "MODE", "WEIGHT" and "WEIGHT_FILE"
2) execute main.py by following command : 
```
    $ python main.py
```
## Todo list
1) Completely parameter valid values check in config.py. Different models may have different valid ranges/values.
2) Modulize train, validation and test function since these are required functions for all models.
3) Make data augmentations more flexible
4) Modulize loss,optimizer, and schedular setting 
5) Full log (tensorboard, summarywriter etc.)
6) Recover config setting by given file
7) Recover train and val set by given file
8) Modify result processor for multi-class classification task, for now, it can only handle binary class