# 鄭敬儒<span style="color:red">_103061240</span>

# Project 4 / Face Detection with a Sliding Window

## Overview
> The goal of this project is to do Face Detection. In training part, we need to find the positive features(face feature) and negative features(NO face feature) by the SIFT-like Histogram of Gradients(HoG) and sliding window to train our classifier. In testing part, we also find the feature in each image by HoG and sliding window, then using the classifier to decide each bounding box in testing image whether it is face or not and record its location.


## Implementation
### Training part
1. Taking face features (get_positive_features.py): 
	* Reading all face image(gray scale, 36* 36), then calculate their HoG to be positive features.
  	  ```
	  img_name = glob.glob(os.path.join(train_path_pos, '*.jpg'))
	  for num in range(len(img_name)):
	    img = imread(img_name[num])
	    pos = hog(img, feature_params['hog_cell_size'])
	    pos_flat = np.reshape(pos, [1, ((feature_params['template_size']/feature_params['hog_cell_size'])**2)*31])
	    if num == 0:
	        features_pos = pos_flat
	    else:
	        features_pos = np.concatenate([features_pos, pos_flat])
  	  ```
2. Taking None face features (get_random_negative_features.py):  
	* Because the None face image are RGB, so we change it to gray scale.
	* For each image, we need to find the 10000/(number of image) feature for each image
	* For each image, I create its pyramid image. The goal of pyramid image is to taking negative examples at **multiple scales image.**
  	  ```
	  for num in range(len(img_name)):
	    img = imread(img_name[num])
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    py = tuple(pyramid_gaussian(gray))
  	  ```
	* Because original or pyramid image may smaller than the features we want to take, I also use some check variable to check it.
	* I choose the bigger three pyramid image (The biggest one is original image). For each, I choose 10000/(number of image) HoG feature randomly. And if the width or height of pyramid image minus 36 is smaller than 10000/(number of image), I will ask it to choose fewer feature in that pyramid image.
  	  ```
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
               new_sample_image = sample_image......
  	  ```
  	  ```
	  ......
	  # If find some image is too small, then only take HoG from that one, ignoring its smaller pyramid image
          if check_size == 1:
             break
  	  ```
	* For each image, I calculate HoG for it and 2 pyramid image.(If some image too small to take 10000/(number of image) out, I will skip the image smaller than it.) From all HoG, I randomly choose 10000/(number of image) or fewer without repeating.
	* Finally, concatenating all image feature together and random choose 10000 from them.
  	  ```
  	  sample_idx = np.random.choice(np.arange(features.shape[0]), size=int(num_samples), replace=False) #replace=False
  	  features_neg = features[sample_idx]
  	  ```
3. Classifier (svm_classify.py):
	* Here using the linear SVM as classifier. And using positive and negitive feature to training it.
  	  ```
  	  clf = svm.LinearSVC()
  	  clf.fit(x,y)
  	  ```

### Testing part
1. Detector (run_detector.py):
	* When detecting, for each image, we detect it by sliding window. And Using (* scale) to change the image size, then using same size sliding window to detect it. For different scale image, same size sliding window can view different Vision.  
  	  ```
  	  min_size = min(img.shape[0], img.shape[1])
  	  scale = 1
  	  
  	  while scale*min_size > feature_params['template_size']:
  	     tmp_img = resize(img, output_shape=[int(img.shape[0]*scale), int(img.shape[1]*scale)]);
  	     hog_feature = hog(tmp_img, feature_params['hog_cell_size'])......
	     
	     ......
	     scale = scale * 0.9
  	  ```
	* Sliding window part to use sliding window to find many BBox feature in testing image
  	  ```
  	  for row in range(hog_feature.shape[0]-cell_num+1):
  	      for col in range(hog_feature.shape[1]-cell_num+1):
  	         hog_seg = hog_feature[row:row+cell_num,col:col+cell_num,:]
  	  ```
	* Using classifier to classifiy each BBox. And if the confidience is too small, then we ignore it, else the confidience is higher than Threshold we calculate it BBox location.
  	  ```
  	  for row in range(hog_feature.shape[0]-cell_num+1):
                for col in range(hog_feature.shape[1]-cell_num+1):
                    hog_seg = hog_feature[row:row+cell_num,col:col+cell_num,:]
                    hog_seg = np.reshape(hog_seg, (1, -1))

                    tmp_confidences = np.reshape(model.decision_function(hog_seg), (1, -1))
                    if tmp_confidences[0,0] > -0.5:
                        cur_confidences = np.concatenate([cur_confidences, tmp_confidences], axis=0)

                        cur_y_min = int(row*cell_size/scale)
                        cur_y_max = int((row+cell_size)*cell_num/scale)
                        cur_x_min = int(col*cell_size/scale)
                        cur_x_max = int((col+cell_size)*cell_num/scale)
  	  ```

### Note
* Because running get_positive_features.py and get_random_negative_features.py take too many time, I store the feature into .npy file.  
	
## Installation
* Other required packages. Install tqdm by conda
* How to compile from source?  I compile in linux terminal on my laptop.

## Results

<center>
<p>
1. This is the Average precision and the Graph precision according to recall.  
<p>
<img src="/code/visualizations/average_precision.png">

2. Here are some detection example
</center>
<table border=1>
<tr>
<td>
<img src="/code/visualizations/detections_Argentina.jpg" width="24%"/>
<img src="/code/visualizations/detections_Arsenal.jpg"  width="24%"/>
<img src="/code/visualizations/detections_audrey2.jpg" width="24%"/>
<img src="/code/visualizations/detections_clapton.jpg" width="24%"/>
</td>
</tr>

<tr>
<td>
<img src="/code/visualizations/detections_cnn2221.jpg" width="24%"/>
<img src="/code/visualizations/detections_original1.jpg"  width="24%"/>
<img src="/code/visualizations/detections_kaari-stef.jpg" width="24%"/>
<img src="/code/visualizations/detections_music-groups-double.jpg" width="24%"/>
</td>
</tr>

</table>


