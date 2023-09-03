# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from os.path import join

######################
# Functions that deal with the result
#   y_score: confidence score (size: [N, class_num])
#   y_label: class number (size: [N])
#   y_pred: predict class number (size: [N])
# TODO : Modify result processor for multi-class classification task, for now, it can only handle binary class
######################
class resultProcessor(object):
    def __init__(self, y_score, y_label, y_pred, save_path, epoch_num):
        self.y_score = y_score
        self.y_label = y_label
        self.y_pred = y_pred
        self.save_path = save_path
        self.epoch_num = epoch_num

    def calConfusionMatrix(self):
        self.cm = confusion_matrix(self.y_label, self.y_pred)
        self.TN, self.FP, self.FN, self.TP = self.cm.ravel()
        
    def getRecall(self):
        return self.TP/(self.TP+self.FN)
    
    def getPrec(self):
        return self.TP/(self.TP+self.FP)
    
    def getAcc(self):
        return (self.TP+self.TN)/(self.TP+self.FP+self.FN+self.TN)
    
    def plotConfusionMatrix(self):
        dispCM = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        dispCM.plot()
        plt.savefig(join(self.save_path, 'epoch_{}_cm.png'.format(self.epoch_num)))
        plt.clf()
    
    def plotROC(self):
        fpr, tpr, thres = roc_curve(self.y_label, self.y_score[:,1])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr,tpr, label="Dog (AUC = {:.2f})".format(auc_score))
        plt.plot([0, 1], [0, 1], "k--", label="(AUC = 0.5)")
        plt.legend()
        plt.savefig(join(self.save_path, 'epoch_{}_ROC.png'.format(self.epoch_num)))
        plt.clf()
        
if __name__ == '__main__':
    import numpy as np
    a = np.array([[0.9, 0.1], 
                  [0.02, 0.98], 
                  [0.13, 0.87], 
                  [0.09, 0.91], 
                  [0.35, 0.65]])
    b = np.array([0,1,1,1,0])
    c = np.array([0,1,1,1,1])
    R = resultProcessor(a, b, c, r".")
    R.calConfusionMatrix()
    R.plotConfusionMatrix()
    R.plotROC()