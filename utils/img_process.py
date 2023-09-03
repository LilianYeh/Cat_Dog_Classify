# -*- coding: utf-8 -*-
from PIL import Image
from torchvision.transforms import Pad
from torchvision import transforms
from os.path import exists, join
from os import makedirs

######################
# Transform tensor to PIL image and save
#   save_path: saving path of converted tensor
#   name: saving same of converted tensor
######################
class TensorToImg(object):
    def __init__(self, save_path):
        self.path = save_path
        if(not exists(save_path)): makedirs(save_path)
        self.transfunc = transforms.ToPILImage()
    def trans(self, img, name="temp.jpg"):
        trans_img = self.transfunc(img)
        trans_img.save(join(self.path, name))
        
######################
# Scale image with its w and h ratio
#   img_size: scale size of longer side
#   img_path: image path
###################### 
class ImgProportionScale(object):
    def __init__(self, img_size):
        self.img_size = img_size
    def __call__(self, img_path):
        img = Image.open(img_path)
        (w,h) = img.size
        
        # resize image, the larger side is resize to self.img_size
        if(w > h): 
            h = int(h*self.img_size/(w*1.0))
            w = self.img_size
        else: 
            w = int(w*self.img_size/(h*1.0))
            h = self.img_size
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        
        return img
    def __repr__(self):
        return self.__class__.__name__+'()'

######################
# Pad image to given size
#   img_size: pad image to img_size x img_size
#   img: PIL image that to be pad
###################### 
class PadImgToGivenSize(object):
    def __init__(self, img_size):
        self.img_size = img_size
        
    def __call__(self, img):
        (w,h) = img.size
        img = Pad(padding=(max(1,(self.img_size-w)//2), max(1,(self.img_size-h)//2)))(img)
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        return img
    def __repr__(self):
        return self.__class__.__name__+'()'

if __name__ == '__main__':
    # test transform
    scaleImg = ImgProportionScale(512)
    pad = PadImgToGivenSize(512)
    I = scaleImg(r'C:\Users\user\Desktop\Interview\aether\HW\data\kaggle_Asirra\train\cat.958.jpg')
    I = pad(I)
    I.save(r'C:\Users\user\Desktop\Interview\aether\HW\temp\cat.958.jpg')
        