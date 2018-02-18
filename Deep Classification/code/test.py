import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from dataloader import *
from model import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

growth_k = 24
nb_block = 2 
#init_learning_rate = 1e-4
epsilon = 1e-4 
dropout_rate = 0.2


train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = color_preprocessing(train_x, test_x)

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])
training_flag = tf.placeholder(tf.bool)
#learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag, dropout_rate=dropout_rate).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_iteration = 10
test_acc = 0.0
test_loss = 0.0
test_pre_index = 0
add = 1000

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            #learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_ / 10.0
        test_acc += acc_ / 10.0

    return test_acc, test_loss

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Loading Fail.")
    
    test_acc, test_loss = Evaluate(sess)
    print "test_loss: %.4f, test_acc: %.4f"%(test_acc, test_loss) 


  
