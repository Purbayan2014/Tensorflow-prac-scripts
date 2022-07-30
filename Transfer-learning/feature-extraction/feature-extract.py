#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 08:58:53 2022

@author: markins
"""

# Transfer learning is using the existing model's 
# architecture proven to work on problems on the current
# situation

# modules
import tensorflow as tf
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow_hub as hub
import datetime
from utils import loss_plotter

# lets set up a callback
"""
Tensorboard callback that tracks the experiment
ModelCheckPoint callback that tracks the Model checkpoint
EarlyStopping Callback that stops the model before it trains too long and overfits
"""

# Experiment tracking
def create_tensorboard_cl(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback =  tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print("Gathering the log data")
    return tensorboard_callback    


# check the gpu
print("Available devices : " , len(tf.config.list_physical_devices("GPU")))


zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip")
zip_ref.extractall()
zip_ref.close()

# check the nos of images
for dirpath, dirnames ,filenames in os.walk("10_food_classes_10_percent"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Load the data
IMAGE_SHAPE = (224,224)
EPOCHS = 5
TRAIN_DIR  = '10_food_classes_10_percent/train'
TEST_DIR = '10_food_classes_10_percent/test'

Train_data_gen = ImageDataGenerator(rescale=1/255.)
Test_data_gen = ImageDataGenerator(rescale=1/255.)

print("Train data")
train_data = Train_data_gen.flow_from_directory(TRAIN_DIR,batch_size=32,
                                                target_size=IMAGE_SHAPE,
                                                class_mode="categorical")

print("Testing data")
test_data = Test_data_gen.flow_from_directory(TEST_DIR,batch_size=32,
                                              target_size=IMAGE_SHAPE,
                                              class_mode="categorical")


# Using 2 pre trained model on tensorflow hub
# Model used - https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1

# https://arxiv.org/abs/1905.11946

# EfficientNet is a convolutional neural network architecture and 
# scaling method that uniformly scales all dimensions of depth/width/resolution 
# using a compound coefficient. Unlike conventional practice that arbitrary scales 
# these factors, the EfficientNet scaling method uniformly scales network width, depth, 
# and resolution with a set of fixed scaling coefficients. For example, if we want to use 
#  times more computational resources, then we can simply increase the network depth by ,
#  width by , and image size by , where  are constant coefficients determined by a small
#  grid search on the original small model. EfficientNet uses a compound coefficient 
#  to uniformly scales network width, depth, and resolution in a principled way.

# In EfficientNet they are scaled in a more principled way i.e. gradually everything 
# is increased.
 
# With considerably fewer numbers of parameters, the family of models are 
# efficient and also provide better results


# BASE EFFICIENT NET MODEL

# The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks 
# of MobileNetV2, in addition to squeeze-and-excitation blocks.



# RESNET-50 ARCHTECTURE

# The main motivation of the ResNet original work was to address the degradation 
# problem in a deep network. Adding more layers to a sufficiently deep neural network 
# would first see saturation in accuracy and then the accuracy degrades

# https://miro.medium.com/max/700/1*5jesyboQzDqOxddUCccqpw.png

# As we can see the training (left) and test errors (right) for the deeper network 
# (56 layer) are higher than the 20 layer network. More the depth and with increasing 
# epochs, the error increases. At first, it appears that as the number of layers increase,
#  the number of parameters increase, thus this is a problem of overfitting. But it is not,

# MATHEMATICAL INTUITION

"""
 lets take a simple DNN architecture including other hyperparameters that can reach 
 a class of function F. so for all f belonging to F there will be some params [w] which 
 we can obtain after training the network for a particular dataset .
 
 Lets take that the ideal function that we need to find is f* but it is not within F,
 then we try to find f1 which is the best case within F. If we design a more powerfull 
 architecture F we should achieve at g1 which is better than f1 but if F doesnt belongs to
 G then there is no guarantee that this assumption will hold.
 
 Conclusion is that if a deeper shallow network func classes contain the simpler and 
 shallower network function classes then we can gurantee that the deeper network will 
 increase the feature finding power of the original network

"""

# RESIDUAL BLOCK

"""
The idea of a residual block is based on the above intiution , a simpler func is a subset 
of a complex func so that the gradient degradation problem can be addresed.

Base activation func = X 
desired mapping input to output func = g(X)

instead of dealing with this fn we deal with a simpler fn g(X)-X , then the original 
mapping is recasted to f(X)+X

https://miro.medium.com/max/700/1*gjoenc3yvXhPMRpoPn4xtA.png

ResNet consists of many residual blocks where residual learning is adopted to every 
few (usually 2 or 3 layers) stacked layers. 

 The building block is shown in Figure 2 and the final output can be considered as y 
 = f(x, W) + x. Here W’s are the weights and these are learned during training. 
 The operation f + x is performed by a shortcut (‘skip’ 2/3 layers) connection and 
 element-wise addition. This is the simplest block where no additional parameters are 
 involved in the skip connection. Element-wise addition is only possible when the 
 dimension of f and x are same, if this is not the case then, we multiply the input
 x by a projection matrix Ws, so that dimensions of f and x matches. In this case the 
 output will change from the previous equation to y = f(x, W) + Ws * x. The elements 
 in the projection matrix will also be trainable.

"""

# RESNET 50
 
"""
In ResNet-50 the stacked layers in the residual block will always have 1×1, 3×3, 
and 1×1 convolution layers. The 1×1 convolution first reduces the dimension and 
then the features are calculated in bottleneck 3×3 layer and then the dimension 
is again increased in the next 1×1 layer. Using 1×1 filter for reducing and increasing 
the dimension of feature maps before and after the bottleneck layer was described in the
 GoogLeNet model by Szegedy et al

"""

# resnet url

resnet_url = "https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/5"

# efficient url

efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"


def create_model(url,classes=10):
    # get the model from the url and save it into a layer
    feature_extractor_layer = hub.KerasLayer(url,
                                             trainable=False,
                                             name="feature_ext_layer",
                                             input_shape=IMAGE_SHAPE+(3,))
    
    # create the sequential model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(classes,activation="softmax",name="output_layer")
        ])
    
    # return the model
    return model

# making the resnet model
resnet_model = create_model(resnet_url,classes=train_data.num_classes)
resnet_model.compile(loss="categorical_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])
res_hs = resnet_model.fit(train_data,
                 steps_per_epoch=len(train_data),
                 validation_data=test_data,
                 validation_steps=len(test_data),
                 epochs=5,
                 callbacks=[create_tensorboard_cl(dir_name="tensorflow_hub", experiment_name="resnet50V2")])



resnet_model.summary()

# Efficient net model
# https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html

eff_model = create_model(efficientnet_url,
                         classes=train_data.num_classes)

eff_model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

eff_hs = eff_model.fit(train_data,
                       epochs=5,
                       steps_per_epoch=len(train_data),
                       validation_data=test_data,
                       validation_steps=len(test_data),
                       callbacks=[create_tensorboard_cl(dir_name="tensorflow_hub", experiment_name="efficientnetB0")])


eff_model.summary()
print(f"The resnetV250 has {len(resnet_model.layers[0].weights)} layers ")
print(f"The efficient-netB0 model has {len(eff_model.layers[0].weights)} layers ")

# create the plot loss
print("Loss curve of the resnet ")
loss_plotter(res_hs)
print("Loss curve of the efficient net")
loss_plotter(eff_hs)

# comparing the results in tensorboard
# tensorboard dev upload --logdir ./tensorflow_hub --name "Resnet50V2 vs EfficientnetB0" --description "Results regarding feature extraction tool" --one_shot
