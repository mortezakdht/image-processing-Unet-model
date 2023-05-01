import numpy as np
from skimage.color import rgb2gray


class Pre_Processing_R2g():
    
    def __init__(self, images):
        self.img = images
        
    def R_2_G(self):
        
        Image_Numbers = self.img.shape[0]         
        Image_Height =  self.img.shape[1]         
        Image_Width =  self.img.shape[2]         

        Image_R_2_G = np.zeros((Image_Numbers, Image_Height, Image_Width), 
                               dtype = np.uint8)

        for i in range(Image_Numbers):
            
       		 Image_R_2_G[i] = rgb2gray(self.img[i]) 

            
        return Image_R_2_G
