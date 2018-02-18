<center>
<img src="README_files/Overview.png" alt="overview" style="float:middle;">
</center>

# Deep Classification

## Brief
* Due: <b>Jan. 6</b>, 11:59pm.
* Required files: results/index.md, and code/


## Overview

Recently, deep learning achieved impressive performance on classification benchmarks such as [MNIST](http://yann.lecun.com/exdb/mnist/), [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html), and [ImageNet](http://www.image-net.org/challenges/LSVRC/) challenges. Starting from AlexNet, the network architecture grows deeper and deeper(Inception, VGG16, ResNet, etc.), and many advanced techniques were introduced (relu, batchnorm, skip connection, bottleneck). In this homework, we are going to implement simple classifier on the Cifar10 dataset, which consists 50000 training and 10000 testing images (32x32 color images) in 10 classes. With many convenient deep learning frameworks ([Tensorflow](https://www.tensorflow.org/), [Pytorch](http://pytorch.org/), [Caffe2](https://caffe2.ai/), and [MXNet](https://mxnet.apache.org/)), implementing deep convolution neural network (CNN) with parallel computing from either CPU or GPU becomes much easier. In this homework, you need to implement a simple data provider for the Cifar10 dataset, a CNN model, and a training and testing process.
     
## Requirement   

- Python
- [TensorFlow](https://www.tensorflow.org/)
- or [Pytorch](http://pytorch.org/)
- or any other framework that may help you implement your CNN model.

## Details and References
In this homework, we won't provide any starter codes, but some advices. Typically, your project will have:
<ul>
    <li><code><font color="green">dataloader.py</font></code>: Contains the dataloader to load the training or testing image and label pairs from the dataset.</li>
    <li><code><font color="green">model.py</font></code>: Contains the CNN network used to perform the classification (You can adopt the architecture of AlexNet, VGG16, etc.</li>
    <li><code><font color="green">train.py</font></code>: Contains the training process including forward, backward, parameter update, saving model, even recording summaries (such as <a href="https://www.tensorflow.org/get_started/summaries_and_tensorboard">tensorboard</a>). </li>
    <li><code><font color="green">test.py</font></code>: Contains the testing process including model restoration, performance evaluation, etc.</li>
</ul>
For the beginners to Tensorflow and Pytorch, you may find the examples of some CNN classifyer quite helpful:

- Tensorflow: [MNIST classification](https://www.tensorflow.org/get_started/mnist/pros)
- Pytorch: [MNIST classification](https://github.com/pytorch/examples/tree/master/mnist) and [more examples](https://l.facebook.com/l.php?u=https%3A%2F%2Fgithub.com%2Fyunjey%2Fpytorch-tutorial%2Ftree%2Fmaster%2Ftutorials%2F01-basics%2Ffeedforward_neural_network&h=ATOBZC-GSOBfyVc8xBqK5lCsQWmLmqDLinggvjmwbDGwfrCofXoP5a2o45csrl0vFUP3rOBzaKSgnpydRtvgItgMAHqL2R5pxSDVJ-JGYNo9kU8OuPPH4dgDI5rj7E_DqwetWg)

## Data

### Introduction

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. The python version dataset can be downloaded from the following [link](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

For simplicity, we also provide the processed training dataset and testing dataset in the following [link](https://drive.google.com/drive/folders/1-G9TyQel2jp51weUrq3yvatGG6PZYVu8?usp=sharing)

### Details of the provided dataset
After downloading the provided training and testing data from the [link](https://drive.google.com/drive/folders/1-G9TyQel2jp51weUrq3yvatGG6PZYVu8?usp=sharing), you can simply load the data by:
```python
import pickle

with open('cifar10_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
```
The train data will be a dictionary containing three types of data with corresponding keys: images, labels, and filenames. For example, <code>train_data['images'][idx]</code>'s label will be <code>train_data['labels'][idx]</code>, where idx (0 to 49999 for training set) is the index of the desired image. Label 0 to 9 corresponds to airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. 



## Writeup
    
You are required to implement a **deep-learning-based method** to recognize the class label of images. Note that many factors such as network architecture, learning rate, batch size, optimizer types may influence the final performance of your model. The purpose of this homework is giving the students chances to train a deep CNN model on their own. Therefore, **as long as you can make your own network work on the training and testing process and report the final performance, any suitable network architecture (AlexNet, VGG16, DenseNet...) is OK (We know that the computation budget varies from person to person, and not everyone has a Titan X to play with.)**
 
For this project, and all other projects, you must do a project report in results folder using [Markdown](https://help.github.com/articles/markdown-basics). We provide you with a placeholder [index.md](./results/index.md) document which you can edit. In the report you will describe your algorithm and any decisions you made to write your algorithm a particular way. For example, your code may consist dataloader.py, model.py, train.py, and test.py. You need to briefly discuss the contents of each file with some code highlights. 
Then, you will describe how to run your code and if your code depended on other packages, please mention the details in the installation part. You also need to show and discuss the results of your algorithm. **Note that the limitation of your computer(talk about the limitation you encountered such as limited RAM, GPU memory, no GPU available...), model selection, training parameter settings (learning rate, batch size, etc.), training process (loss), testing result (accuracy and confusion matrix) are important parts that required to be included in your report**. Discuss any extra credit you did, and clearly show what contribution it had on the results (e.g. performance with and without each extra credit component).


## Rubric
<ul>
   <li> 50 pts: Perform training on the cifar10 dataset. </li>
	<li> 30 pts: Evaluating testing result on the testing set.  </li>
   <li> 20 pts: Write up reports with Overview, Implementation, Installation, Results(with discussion on the training and testing processes mentioned in Writeup) </li>
   <li> 10 pts: Extra credit (up to ten points).</li>
   <li> -5*n pts: Lose 5 points for every time (after the first) you do not follow the instructions for the hand in format </li>
</ul> 

## Get start & hand in
* Publicly fork version (+2 extra points)
	- [Fork the homework](https://education.github.com/guide/forks) to obtain a copy of the homework in your github account
	- [Clone the homework](http://gitref.org/creating/#clone) to your local space and work on the code locally
	- Commit and push your local code to your github repo
	- Once you are done, submit your homework by [creating a pull request](https://help.github.com/articles/creating-a-pull-request)

* [Privately duplicated version](https://help.github.com/articles/duplicating-a-repository)
  - Make a bare clone
  - mirror-push to new repo
  - [make new repo private](https://help.github.com/articles/making-a-private-repository-public)
  - [add aliensunmin as collaborator](https://help.github.com/articles/adding-collaborators-to-a-personal-repository)
  - [Clone the homework](http://gitref.org/creating/#clone) to your local space and work on the code locally
  - Commit and push your local code to your github repo
  - I will clone your repo after the due date

