# -*- coding: utf-8 -*-
from torchvision.models import resnet50
from torch.nn import Linear
from os.path import join
from torch.nn import CrossEntropyLoss, Softmax
from torch import optim
import torch
from torch.autograd import Variable
from sys import stdout
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.optim import lr_scheduler # import CyclicLR
import csv
from os import sep as osSep
from utils.result_process import resultProcessor
from torchsummary import summary
class ResNet50():
    def __init__(self, config):
        self.mode = config.MODE
        self.weight = config.WEIGHT
        self.class_num = config.CLASSES
        self.cuda = config.CUDA
        self.save_path = join(config.OUTPUT_DIR, "model")
        self.weight_file = config.WEIGHT_FILE
        self.lr = config.LR
        self.epoch = config.EPOCH
        self.batch = config.BATCH
        self.tensor_type = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        self.opt = config.OPTIMIZER
        self.sche = config.SCHEDULAR
        self.save_root = config.OUTPUT_DIR
    
    def build(self):
        print("Create ResNet50 from {}...".format(self.weight))
        if(self.weight != "FILE"):
            model = resnet50(weights=self.weight)
            num_feature = model.fc.in_features
            model.fc = Linear(num_feature,self.class_num)
        else:
            model = resnet50()
            num_feature = model.fc.in_features
            model.fc = Linear(num_feature,self.class_num)
            checkpoint = torch.load(self.weight_file)
            model.load_state_dict(checkpoint['state_dict'])
        if(self.cuda): model = model.cuda()
        self.model = model
    
    # show model architecture and feature map size by given input
    def showModel(self, input_size):
        print('---------------- Model ----------------')
        summary(self.model, (3, input_size, input_size))
        print('---------------- Model ----------------')
        
    def train(self, train_loader, val_loader):
        print("Training...")
        # train 
        # TODO : Modulize loss,optimizer, and schedular setting  
        criterion = CrossEntropyLoss()
        
        # get optimizer by name
        optimizer = getattr(optim, self.opt)
        optimizer = optimizer(self.model.parameters(), lr=self.lr)
        
        # get schedular by name
        if(self.sche):
            batch_size = len(train_loader)
            scheduler = getattr(lr_scheduler, self.sche)
            scheduler = scheduler(optimizer, base_lr=self.lr*(1e-2), max_lr=self.lr, cycle_momentum=False, step_size_up=batch_size*5)

        
        # TODO : Full log (tensorboard, summarywriter etc.)
        best_loss = 100
        
        for epoch in range(self.epoch):
            self.model.train()
            total_loss = 0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                img = Variable(batch["img"].type(self.tensor_type))
                label= batch["label"].cuda()
                # forward
                output = self.model(img)
                # backward
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                if(self.sche): scheduler.step()
                
                total_loss += loss.item()
                stdout.write('\rBatch {}/{} : loss {:.4f}\r'.format(i, batch_size, loss.item()))
                #if(i > 50): break
            print("\nEpoch{}/{} : loss {:.4f}".format(epoch, self.epoch, total_loss/batch_size))
            
            # perform validation, if validation result is better than save the result
            result, val_loss = self.val(val_loader, criterion, epoch_num=epoch)
            if(val_loss < best_loss):
                best_loss = val_loss
                state = {'epoch': epoch, 
                         'state_dict': self.model.state_dict(),
                         'loss': total_loss/batch_size,
                         'val_loss': val_loss,
                         'val_recall': result.getRecall(),
                         'val_acc': result.getAcc()}
                result.plotConfusionMatrix()
                result.plotROC()
                torch.save(state, join(self.save_path, 'epoch_{}.pth.tar'.format(epoch)))
    
    def val(self, val_loader, criterion, epoch_num=0):
        print("Validation...")
        self.model.eval()
        cal_confidence = Softmax(dim=1)
        with torch.no_grad():
            total_loss = 0
            y_pred = np.array([])
            y_true = np.array([])
            y_score = np.array([])
            val_iter = iter(val_loader)
            for i in range(len(val_iter)):
                stdout.write('\rProcessing Batch {}/{} \r'.format(i+1, len(val_iter)))
                imgs = val_iter.next()
                img, label = imgs["img"].cuda(), imgs["label"].cuda(),
                output  = self.model(img)
                loss = criterion(output, label)
                total_loss += loss.item()
                y_pred = np.concatenate((y_pred,torch.argmax(output, 1, keepdim=True).cpu().numpy().flatten()), axis = 0)
                y_true = np.concatenate((y_true,torch.argmax(label, 1, keepdim=True).cpu().numpy().flatten()), axis = 0)
                if(len(y_score) == 0): y_score = cal_confidence(output).cpu().numpy()
                else: y_score = np.concatenate((y_score, cal_confidence(output).cpu().numpy()), axis = 0)
            result = resultProcessor(y_score, y_true, y_pred, self.save_path, epoch_num)
            result.calConfusionMatrix()
            print("\nval result: \n\t{:10}: {:.4f}\n\t{:10}: {:.4f}\n\t{:10}: {:.4f}".format("Recall", result.getRecall(), "Precision", result.getPrec(), "Accuracy", result.getAcc()))
            total_loss = total_loss/len(val_iter)
            print('\t{:10}: {:.4f}'.format("Loss", total_loss))
            return result, total_loss
    def test(self, test_loader):
        print("Testing...")
        csv_file = open(join(self.save_root, 'out.csv'), 'w', newline="") 
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["id", "Dog", "Dog score"])
        
        cal_confidence = Softmax(dim=1)
            
        self.model.eval()
        with torch.no_grad():
            test_iter = iter(test_loader)
            for i in range(len(test_iter)):
                stdout.write('\rProcessing Batch {}/{} \r'.format(i+1, len(test_iter)))
                imgs = test_iter.next()
                img = imgs["img"].cuda()
                output  = self.model(img)
                confidence = cal_confidence(output)
                
                y_pred = torch.argmax(output, 1, keepdim=True).cpu().numpy().flatten()
                confidence = confidence.cpu().numpy()
                for idx in range(len(y_pred)):
                    name = int(imgs["name"][idx].split(osSep)[-1].split('.')[0])
                    csv_writer.writerow([name, y_pred[idx], confidence[idx][1]])
                #if(i>50): break
        csv_file.close()
                
                
                
            
            
            

 