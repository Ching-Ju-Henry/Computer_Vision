import numpy as np
import os
import random
from cyvlfeat.hog import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import color
from tqdm import tqdm
import cv2
import glob
import pdb
import math

# you may implement your own data augmentation functions

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    '''
    FUNC: This funciton should return negative training examples (non-faces) from
        any images in 'non_face_scn_path'. Images should be converted to grayscale,
        because the positive training data is only available in grayscale. For best
        performance, you should sample random negative examples at multiple scales.
    ARG:
        - non_face_scn_path: a string; directory contains many images which have no
                             faces in them.
        - feature_params: a dict; with keys,
                          > template_size: int (probably 36); the number of
                            pixels spanned by each train/test template.
                          > hog_cell_size: int (default 6); the number of pixels
                            in each HoG cell. 
                          Template size should be evenly divisible by hog_cell_size.
                          Smaller HoG cell sizez tend to work better, but they 
                          make things slower because the feature dimenionality 
                          increases and more importantly the step size of the 
                          classifier decreases at test time.
    RET:
        - features_neg: (N,D) ndarray; N is the number of non-faces and D is 
                        the template dimensionality, which would be, 
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
        - neg_examples: TODO
    '''
    #########################################
    ##          you code here              ##
    #########################################
    
    #reading image
    img_name = glob.glob(os.path.join(non_face_scn_path, '*.jpg'))
    neg_examples = len(img_name)
    sample_image = math.ceil(num_samples/neg_examples)
    features = np.zeros([1, int((feature_params['template_size'] / feature_params['hog_cell_size'])**2 * 31)])
    
    #each image
    for num in range(len(img_name)):
          img = imread(img_name[num])
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          py = tuple(pyramid_gaussian(gray))
          #each pyramid image (the bigger three)
          check_dif_py = 0
          check_size = 0 #pyramid image may too small, use this one to check
          for ch in range(0,3):
             w = py[ch].shape[1]
             h = py[ch].shape[0]
             #pyramid image or original image may too small to take 36 feature, so using this to check
             if min(w-feature_params['template_size'],h-feature_params['template_size']) <= sample_image:
               new_sample_image = min(w-feature_params['template_size'],h-feature_params['template_size'])
               check_size = check_size + 1
             else:
               new_sample_image = sample_image
             #choose BBox in each pyramid image
             row_base = (np.random.choice(np.arange(h-feature_params['template_size']), size=int(new_sample_image), replace=False))
             col_base = (np.random.choice(np.arange(w-feature_params['template_size']), size=int(new_sample_image), replace=False))
             #take feature from each pyramid image
             check_same_py = 0
             for f_idx in range (int(new_sample_image)):
                 img_section = img[row_base[f_idx]:row_base[f_idx]+feature_params['template_size'], col_base[f_idx]:col_base[f_idx]+feature_params['template_size']]
                 neg = hog(img_section, feature_params['hog_cell_size'])
                 neg_flat = np.reshape(neg, [1, ((feature_params['template_size']/feature_params['hog_cell_size'])**2)*31])
                 if check_same_py == 0:
                     this_py = neg_flat
                     check_same_py =check_same_py + 1
                 else:
                     this_py = np.concatenate([this_py, neg_flat])
             if check_dif_py == 0:
                all_py = this_py
                check_dif_py = check_dif_py + 1
             else:
                all_py = np.concatenate([all_py, this_py])
             if check_size == 1:
               break
          if check_size == 0:
              feature_this_img = np.random.choice(np.arange(all_py.shape[0]), size=int(new_sample_image)*2, replace=False)#
          else:
              feature_this_img = np.random.choice(np.arange(all_py.shape[0]), size=int(new_sample_image), replace=False)
          x = 0
          
          for le in range(feature_this_img.shape[0]):
             if x == 0:
                tmp = np.reshape(all_py[feature_this_img[le]], [1, ((feature_params['template_size']/feature_params['hog_cell_size'])**2)*31])
                x = x +1
             else:
                tmp = np.concatenate([tmp, np.reshape(all_py[feature_this_img[le]], [1, ((feature_params['template_size']/feature_params['hog_cell_size'])**2)*31])])
          
          features = np.concatenate([features, tmp])
    print features.shape
    sample_idx = np.random.choice(np.arange(features.shape[0]), size=int(num_samples), replace=False) #replace=False
    features_neg = features[sample_idx]
    
    
    #np.save("features_neg_my.npy", features_neg)
    #np.save("neg_examples.npy", neg_examples)
    #########################################
    ##          you code here              ##
    #########################################
            
    return features_neg, neg_examples

