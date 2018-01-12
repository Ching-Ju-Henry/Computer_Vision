# 鄭敬儒<span style="color:red">_103061240</span>

# Project 3 / Scene recognition with bag of words

## Overview
> The goal of this project is to do image recognition. We need to implement two different way of feature extracting and classifiers. In feature extracting part, we use method of tiny images and nearest neighbor classification(KNN, K=1). In classifier part, we use bags of SIFT words and SVM. Finally, we assemble these method in different combination to observe the accuracy. 

## Implementation
### Feature Extracting
1. **Tiny images feature: (get_tiny_images.py)**
	* Method: Resizing the image smaller, here I resize images into 16* 16. Then Normalizing the image (minus mean, then division standard deviation). Finally, flatting each image into 1* 256 array
  
  	  ```
	  width  = 16
	  height = 16
	  
	  tiny_images = np.zeros((len(image_paths), width*height), dtype='float32')
	  
	  for num in range(len(image_paths)):
	      img = Image.open(image_paths[num])
	      img_resize = np.asarray(img.resize((width, height), Image.ANTIALIAS), dtype='float32')
	      img_resize = img_resize.flatten()
	      img_tiny = (img_resize-np.mean(img_resize))/np.std(img_resize)
	      tiny_images[num, :] = img_tiny
        
	  return tiny_images
  	  ```

2. **Bag of SIFT feature: (get_bags_of_sifts.py and build_vocabulary.py)**
	* Step1 (build_vocabulary.py): At first, we need to build vocabularies from training and testing images by K-means method, then we get three vocabularies files (.pkl). 
	* Step2 (get_bags_of_sifts.py): By this way, we can get the SIFT feature, then classify each feature into nearest cluster centers. Finally, calculating amount of feature in each cluster.
	
  	  ```
	  f = open('vocab.pkl', 'rb')
	  voc = pickle.load(f)
	  voc_size = len(voc)
	  len_img = len(image_paths)
	  image_feats = np.zeros((len_img, voc_size))
	  
	  for idx, path in enumerate(image_paths):
	      img = np.asarray(Image.open(path),dtype='float32')
	      frames, descriptors = dsift(img, step=[5,5], fast=True)
	      d = distance.cdist(voc, descriptors, 'euclidean')
	      dist = np.argmin(d, axis = 0)
	      histo, bins = np.histogram(dist, range(voc_size+1))
	      norm = np.linalg.norm(histo)
	      if norm == 0: 
	          image_feats[idx, :] = histo
	      else:
	          image_feats[idx, :] = histo/norm
	  return image_feats
  	  ```	

### Classifier
1. **Nearest neighbor classification: (nearest_neighbor_classify.py)**
	* Method: Here I calculate the distance between each testing images and all training images, then I choose the min distance for each testing images to decide which class it belonging.
	
  	  ```
	  test_predicts = []
	  for num in range(test_image_feats.shape[0]):
	      each_row = []
	      each = np.tile(test_image_feats[num],(train_image_feats.shape[0],1))
	      squ_each = np.square(each - train_image_feats)
	      for sq in range(squ_each.shape[0]):
	          each_row.append(np.sqrt(sum(squ_each[sq])))
	      smallest = min(each_row)
	      smallest_ind = each_row.index(min(each_row))
	      test_predicts.append(train_labels[smallest_ind])
	      
	  return test_predicts
  	  ```

2. **SVM classification: (svm_classify.py)**
	* Method: Here I use the 'sklearn' to do linear SVM. (LinearSVC() in sklearn.svm)
	
  	  ```
	  clf = LinearSVC()
	  clf.fit(train_image_feats, train_labels)
	  pred_label = clf.predict(test_image_feats)
	  
	  return pred_label
  	  ```

### [Extra]
1. **Non-linear SVM classification**
	* Here I also do non-linear SVM by sklearn (SVC in sklearn.svm)
	
  	  ```
	  clf = SVC(kernel = 'rbf', random_state=0, gamma = 0.2, C=10)
	  clf.fit(train_image_feats, train_labels)
	  pred_label = clf.predict(test_image_feats)
	  
	  return pred_label
  	  ```
	
2. **Cross Validation**
	* Here I calculate Validation Score by (cross_val_score in sklearn.model_selection)
	
  	  ```
	  sc = cross_val_score(clf, train_image_feats, train_labels)
	  print 'Validation Score:',sc.mean()
  	  ```
	  
## Results
| Feature | Classifier | Accuracy | confusion matrix |
| ------- | ---------- | -------- | ---------------- |
Tiny images | Nearest neighbor | 0.2313 | ![](result_bag+near.png)  
Bag of SIFT | Nearest neighbor | 0.5533 | ![](result_bag+nearest.png)  
Bag of SIFT | Linear SVM | 0.7106 | ![](result_bag+linearsvm.png)  
Bag of SIFT | Non-Linear SVM | 0.7153 | ![](result_best.png)  

### Visualization (Best one: Bag of SIFT + Non-linear SVM)
| Category name | Accuracy |Sample training images | Sample true positives | False positives with true label | False negatives with wrong predicted label |
| :-----------: | :------: |:--------------------: | :-------------------: | :-----------------------------: | :----------------------------------------: |
| Kitchen | 0.63|![](thumbnails/Kitchen_train_image_0205.jpg) | ![](thumbnails/Kitchen_TP_image_0088.jpg) | ![](thumbnails/Kitchen_FP_image_0287.jpg) | ![](thumbnails/Kitchen_FN_image_0072.jpg) |
| Store | 0.61|![](thumbnails/Store_train_image_0110.jpg) | ![](thumbnails/Store_TP_image_0248.jpg) | ![](thumbnails/Store_FP_image_0061.jpg) | ![](thumbnails/Store_FN_image_0099.jpg) |
| Bedroom | 0.55|![](thumbnails/Bedroom_train_image_0110.jpg) | ![](thumbnails/Bedroom_TP_image_0088.jpg) | ![](thumbnails/Bedroom_FP_image_0230.jpg) | ![](thumbnails/Bedroom_FN_image_0157.jpg) |
| LivingRoom | 0.51|![](thumbnails/LivingRoom_train_image_0035.jpg) | ![](thumbnails/LivingRoom_TP_image_0177.jpg) | ![](thumbnails/LivingRoom_FP_image_0261.jpg) | ![](thumbnails/LivingRoom_FN_image_0176.jpg) |
| Office | 0.91|![](thumbnails/Office_train_image_0035.jpg) | ![](thumbnails/Office_TP_image_0103.jpg) | ![](thumbnails/Office_FP_image_0035.jpg) | ![](thumbnails/Office_FN_image_0159.jpg) |
| Industrial | 0.65|![](thumbnails/Industrial_train_image_0289.jpg) | ![](thumbnails/Industrial_TP_image_0237.jpg) | ![](thumbnails/Industrial_FP_image_0216.jpg) | ![](thumbnails/Industrial_FN_image_0277.jpg) |
| Suburb | 0.94|![](thumbnails/Suburb_train_image_0110.jpg) | ![](thumbnails/Suburb_TP_image_0065.jpg) | ![](thumbnails/Suburb_FP_image_0147.jpg) | ![](thumbnails/Suburb_FN_image_0234.jpg) |
| InsideCity | 0.64|![](thumbnails/InsideCity_train_image_0294.jpg) | ![](thumbnails/InsideCity_TP_image_0240.jpg) | ![](thumbnails/InsideCity_FP_image_0087.jpg) | ![](thumbnails/InsideCity_FN_image_0277.jpg) |
| TallBuilding | 0.67|![](thumbnails/TallBuilding_train_image_0110.jpg) | ![](thumbnails/TallBuilding_TP_image_0116.jpg) | ![](thumbnails/TallBuilding_FP_image_0279.jpg) | ![](thumbnails/TallBuilding_FN_image_0199.jpg) |
| Street | 0.68|![](thumbnails/Street_train_image_0110.jpg) | ![](thumbnails/Street_TP_image_0268.jpg) | ![](thumbnails/Street_FP_image_0199.jpg) | ![](thumbnails/Street_FN_image_0177.jpg) |
| Highway | 0.8|![](thumbnails/Highway_train_image_0097.jpg) | ![](thumbnails/Highway_TP_image_0252.jpg) | ![](thumbnails/Highway_FP_image_0183.jpg) | ![](thumbnails/Highway_FN_image_0140.jpg) |
| OpenCountry | 0.63|![](thumbnails/OpenCountry_train_image_0360.jpg) | ![](thumbnails/OpenCountry_TP_image_0395.jpg) | ![](thumbnails/OpenCountry_FP_image_0275.jpg) | ![](thumbnails/OpenCountry_FN_image_0029.jpg) |
| Coast | 0.78|![](thumbnails/Coast_train_image_0345.jpg) | ![](thumbnails/Coast_TP_image_0282.jpg) | ![](thumbnails/Coast_FP_image_0064.jpg) | ![](thumbnails/Coast_FN_image_0063.jpg) |
| Mountain | 0.8|![](thumbnails/Mountain_train_image_0035.jpg) | ![](thumbnails/Mountain_TP_image_0351.jpg) | ![](thumbnails/Mountain_FP_image_0212.jpg) | ![](thumbnails/Mountain_FN_image_0369.jpg) |
| Forest | 0.93|![](thumbnails/Forest_train_image_0097.jpg) | ![](thumbnails/Forest_TP_image_0102.jpg) | ![](thumbnails/Forest_FP_image_0219.jpg) | ![](thumbnails/Forest_FN_image_0275.jpg) |


