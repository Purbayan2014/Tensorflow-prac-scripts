#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 08:04:44 2022

       /\         markins@archcraft
      /  \        os     Archcraft
     /\   \       host   G3 3579
    /      \      kernel 5.18.14-arch1-1
   /   ,,   \     uptime 45m
  /   |  |  -\    pkgs   1558
 /_-''    ''-_\   memory 2062M / 15844M

@author: markins
"""

""" 

So in feature extracting some of the layer stays frozen we are not able to fine tune the efficient net model or the resnet model that we are using 
But in fine tuning we are able to unfreeze the layers in the resnet or the efficient model that's maybe the top 10 layers or maybe all of the layers
and we are able to train the model on the existing data , 

NB : FOR FINE TUNING WE WILL ALWAYS REQUIRE SOME EXTRA DATA

"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.activations import softmax
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from utils import loss_plotter, create_tensorboard_callback, unzip_data, walker_dt_dir

# check the gpu
print("Available devices : " , len(tf.config.list_physical_devices("GPU")))

# !nvidia-smi

# unzip_data("10_food_classes_10_percent.zip")
walker_dt_dir("10_food_classes_10_percent")

train_path = "10_food_classes_10_percent/train"
test_path = "10_food_classes_10_percent/test"

IMG_SIZE = (224,224)
BATCH_SIZE = 32


# Much faster than ImageDataGenerator
train_dt_10  = image_dataset_from_directory(directory=train_path,
                                            image_size=IMG_SIZE,
                                            label_mode="categorical",
                                            batch_size=BATCH_SIZE)

test_dt_10 = image_dataset_from_directory(directory=test_path,
                                          image_size=IMG_SIZE,
                                          label_mode="categorical",
                                          batch_size=32)

"""
Found 750 files belonging to 10 classes.
Found 2500 files belonging to 10 classes.
<BatchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 10), dtype=tf.float32, name=None))>

Nos of total samples  = 750 and 75 per class
Nos of classes = 10
Batch size = 32
Image size = heigth and width
Nos of color channels = red, green and blue i:e = 3
Nos of classes in the label tensors (10 types of food)
"""

# train_dt_10 

# train_dt_10.class_names

"""
['chicken_curry',
 'chicken_wings',
 'fried_rice',
 'grilled_salmon',
 'hamburger',
 'ice_cream',
 'pizza',
 'ramen',
 'steak',
 'sushi']
"""

# an eg of batch dataset
for images, labels in train_dt_10.take(1):
    print(images, labels)

"""
tf.Tensor(
[[[[159.35715   141.35715   129.35715  ]
   [158.66837   140.66837   128.66837  ]
   [158.71939   140.71939   128.71939  ]
   ...
   [152.92851   130.92851   117.928505 ]
   [155.28558   133.28558   120.28558  ]
   [133.51526   111.51526    98.51526  ]]

  [[161.28572   143.28572   131.28572  ]
   [160.92857   142.92857   130.92857  ]
   [161.92857   143.92857   131.92857  ]
   ...
   [151.301     132.301     118.300995 ]
   [148.29065   129.29065   115.29065  ]
   [131.11244   112.11244    98.11244  ]]

  [[162.64285   144.64285   132.64285  ]
   [160.71428   142.71428   130.71428  ]
   [160.42857   142.42857   130.42857  ]
   ...
   [152.68872   133.68872   119.68873  ]
   [143.4284    124.4284    110.4284   ]
   [139.021     120.020996  106.020996 ]]

  ...

  [[250.50511   242.50511   231.50511  ]
   [251.82655   243.82655   232.82655  ]
   [251.33672   243.33672   232.33672  ]
   ...
   [234.40804   219.9795    206.88768  ]
   [231.84175   217.84175   204.84175  ]
   [231.99997   217.99997   206.57144  ]]

  [[249.76021   241.76021   230.76021  ]
   [249.13779   241.13779   230.13779  ]
   [250.84186   242.84186   231.84186  ]
   ...
   [236.51521   222.08669   209.08669  ]
   [237.85712   223.85712   212.85712  ]
   [235.14276   221.14276   210.14276  ]]

  [[243.04561   235.04561   224.04561  ]
   [247.47935   239.47935   228.47935  ]
   [242.79555   234.79555   223.79555  ]
   ...
   [200.93709   186.50856   175.08003  ]
   [197.28409   183.28409   172.28409  ]
   [197.6974    183.6974    172.6974   ]]]


 [[[254.        254.        254.       ]
   [253.        255.        254.       ]
   [252.57143   255.        254.       ]
   ...
   [254.        254.        254.       ]
   [254.        254.        254.       ]
   [254.        254.        254.       ]]

  [[254.        254.        254.       ]
   [253.        255.        254.       ]
   [252.57143   255.        254.       ]
   ...
   [254.        254.        254.       ]
   [254.        254.        254.       ]
   [254.        254.        254.       ]]

  [[254.        254.        254.       ]
   [253.        255.        254.       ]
   [252.66327   255.        254.       ]
   ...
   [254.        254.        254.       ]
   [254.        254.        254.       ]
   [254.        254.        254.       ]]

  ...

  [[253.        253.        253.       ]
   [248.2857    248.2857    248.2857   ]
   [235.85715   235.85715   235.85715  ]
   ...
   [254.        254.        254.       ]
   [254.        254.        254.       ]
   [254.        254.        254.       ]]

  [[253.        253.        253.       ]
   [247.35205   247.35205   247.35205  ]
   [234.7857    234.7857    234.7857   ]
   ...
   [254.        254.        254.       ]
   [254.        254.        254.       ]
   [254.        254.        254.       ]]

  [[253.        253.        253.       ]
   [247.2857    247.2857    247.2857   ]
   [233.78061   233.78061   233.78061  ]
   ...
   [254.        254.        254.       ]
   [254.        254.        254.       ]
   [254.        254.        254.       ]]]


 [[[183.2143    168.2143    125.21428  ]
   [182.20409   167.20409   124.20408  ]
   [181.54082   166.54082   123.54082  ]
   ...
   [ 93.959175   52.67346    54.13265  ]
   [ 94.92857    52.908184   53.70411  ]
   [ 92.11223    51.19387    50.18366  ]]

  [[180.07143   165.07143   122.07143  ]
   [180.07143   165.07143   122.07143  ]
   [180.73468   165.73468   122.734695 ]
   ...
   [110.40814    58.82653    58.163258 ]
   [112.54082    58.887756   58.928574 ]
   [110.79591    57.204075   55.499985 ]]

  [[180.21428   165.42857   122.       ]
   [180.7347    165.94899   122.52041  ]
   [182.70409   167.91837   124.4898   ]
   ...
   [113.98978    64.42856    62.857143 ]
   [116.591805   65.530624   62.979588 ]
   [115.642845   63.57142    61.357136 ]]

  ...

  [[194.22441   204.22441   195.22441  ]
   [197.86732   207.86732   199.2959   ]
   [200.90822   210.90822   202.90822  ]
   ...
   [214.33676   209.90823   175.1225   ]
   [214.03055   208.70406   174.3673   ]
   [215.19376   209.19376   175.19376  ]]

  [[200.13264   210.13264   201.13264  ]
   [201.35713   211.35713   202.7857   ]
   [201.87753   211.87753   203.87753  ]
   ...
   [224.03055   220.03055   185.03055  ]
   [218.48994   214.48994   179.48994  ]
   [209.33676   205.33676   170.33676  ]]

  [[199.14285   209.14285   200.14285  ]
   [200.30615   210.30615   201.73473  ]
   [200.45923   210.45923   202.45923  ]
   ...
   [211.44887   207.44887   172.44887  ]
   [212.06104   208.06104   173.06104  ]
   [212.52026   208.52026   173.52026  ]]]


 ...


 [[[183.35713   208.2857    224.35713  ]
   [182.5       205.5       223.5      ]
   [185.70918   207.71939   226.64285  ]
   ...
   [156.11212   182.0407    191.39783  ]
   [158.45918   182.45918   192.45918  ]
   [156.8674    180.1531    190.51025  ]]

  [[185.35715   208.2143    226.42857  ]
   [185.28061   206.28061   225.42346  ]
   [187.81122   208.        227.81122  ]
   ...
   [161.29083   185.57654   197.50511  ]
   [162.87239   187.1581    199.08667  ]
   [157.35732   181.64304   193.57161  ]]

  [[188.64287   207.28572   229.07144  ]
   [188.69897   207.34183   229.12755  ]
   [190.95407   208.61736   230.78572  ]
   ...
   [154.74474   186.16826   194.19376  ]
   [155.38266   187.31123   195.16837  ]
   [149.76521   181.69377   189.55092  ]]

  ...

  [[ 43.510204   29.510202   16.510202 ]
   [ 46.44385    33.44385    17.443851 ]
   [ 39.57141    26.571411   10.357125 ]
   ...
   [ 26.862253   22.862253   23.862253 ]
   [ 23.98472    21.98472    22.98472  ]
   [ 20.91833    18.91833    19.91833  ]]

  [[ 40.831585   26.831587   13.831587 ]
   [ 41.224434   28.224434   12.224435 ]
   [ 39.857178   26.857178   10.642892 ]
   ...
   [ 26.556171   22.556171   23.556171 ]
   [ 29.846973   27.846973   28.846973 ]
   [ 24.714233   22.714233   23.714233 ]]

  [[ 39.642853   25.642853   12.642853 ]
   [ 42.09693    29.096931   13.096931 ]
   [ 45.438763   32.438763   16.224478 ]
   ...
   [ 26.183527   22.183527   23.183527 ]
   [ 23.020472   21.020472   22.020472 ]
   [ 27.102016   25.102016   26.102016 ]]]


 [[[ 28.484694   23.484694   17.484694 ]
   [ 22.545918   19.545918   14.5459175]
   [ 16.714287   13.357142   10.571428 ]
   ...
   [ 81.214264   69.785736   76.       ]
   [ 81.         70.         76.       ]
   [ 80.64282    69.64282    75.64282  ]]

  [[ 27.714285   22.714285   16.714285 ]
   [ 20.285713   17.285713   12.285713 ]
   [ 15.428571   12.071428    9.285714 ]
   ...
   [ 80.28569    68.85716    75.07143  ]
   [ 80.07143    69.07143    75.07143  ]
   [ 80.40309    69.40309    75.40309  ]]

  [[ 27.92857    22.92857    18.92857  ]
   [ 19.285713   16.071426   11.499999 ]
   [ 14.785714   11.428572    8.979592 ]
   ...
   [ 79.42853    68.04591    73.78569  ]
   [ 78.801025   67.801025   73.37245  ]
   [ 80.35718    69.35718    74.928604 ]]

  ...

  [[ 78.85723    58.15311    49.86226  ]
   [ 74.50006    56.571472   47.071426 ]
   [ 71.61741    54.785713   45.55101  ]
   ...
   [ 12.811147    8.260173    7.       ]
   [ 11.          7.          6.       ]
   [  8.07135     6.4285583   4.6428223]]

  [[ 84.209206   59.         49.40307  ]
   [ 80.49495    57.209206   49.346977 ]
   [ 76.14292    55.928596   47.00004  ]
   ...
   [ 13.57135     9.357086    7.9285583]
   [ 10.790779    9.790779    7.790779 ]
   [  8.571381    7.5713806   5.5713806]]

  [[ 84.51526    56.341743   47.9285   ]
   [ 84.37758    59.1683     49.571396 ]
   [ 82.28064    58.71936    51.       ]
   ...
   [ 12.9999695   8.785706    7.3571777]
   [  9.688838    8.688838    6.688838 ]
   [  8.          7.          5.       ]]]


 [[[124.43367   135.78061   150.44897  ]
   [196.88266   190.29082   192.42857  ]
   [172.51021   153.87245   142.7347   ]
   ...
   [123.64797    86.64797    42.142857 ]
   [128.14288    92.14288    43.94897  ]
   [130.86732    95.2245     42.510147 ]]

  [[108.20408   125.39286   150.63776  ]
   [159.92857   158.35715   169.4949   ]
   [177.39795   163.92857   156.7143   ]
   ...
   [119.92857    81.71431    37.142838 ]
   [122.22448    84.22448    37.081593 ]
   [117.33153    80.33153    28.331524 ]]

  [[114.16837   137.4643    179.5204   ]
   [131.79593   139.4796    163.42346  ]
   [161.91837   158.89795   158.65816  ]
   ...
   [118.35714    80.18879    35.525494 ]
   [114.35717    76.35717    29.214287 ]
   [123.418564   85.6992     35.857285 ]]

  ...

  [[  7.494884    7.494884    7.494884 ]
   [  7.9438763   7.9438763   7.9438763]
   [  9.57145     9.57145     9.57145  ]
   ...
   [229.        213.21426   196.57147  ]
   [227.        211.        195.       ]
   [228.8521    212.8521    196.8521   ]]

  [[  9.331628    9.331628    9.331628 ]
   [ 10.85713    10.85713    10.85713  ]
   [ 12.642831   12.642831   12.642831 ]
   ...
   [226.92856   211.14282   194.50003  ]
   [225.93365   209.93365   193.93365  ]
   [228.68884   212.68884   196.68884  ]]

  [[  8.586757    8.586757    8.586757 ]
   [  8.928572    8.928572    8.928572 ]
   [  9.709176    9.709176    9.709176 ]
   ...
   [225.92348   210.13774   193.49495  ]
   [225.07144   209.07144   193.07144  ]
   [228.35718   212.35718   196.35718  ]]]], shape=(32, 224, 224, 3), dtype=float32) tf.Tensor(
[[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]], shape=(32, 10), dtype=float32)
"""

# TODO : Models to be implemented

# model_exp = {
#     "Experiments" : ["Model0(Baseline)","Model 1","Model 2","Model 3","Model 4" ],
#     "Data" : ["10% of data", "1% of the data", "Same as model 0", "Same as model 0", "100% of the training data"],
#     "Preprocessing" : ["None", ["Random flip","Rotation","Zoom","Height","Width","Data Augmentation"],"Same as model 1","Same as model 1", "Same as model 1"],
#     "Model" : ["EfficientNETB0","Same as model 0","Same as model 0", "Fine tuning model trained on EfficientNETB0 with top 10 layers unfrozen","Same as Model 3"]
#     } 

# data = pd.DataFrame.from_dict(model_exp)
# data
"""
                # Modelling experiments
                
|index|Experiments|Data|Preprocessing|Model|
|---|---|---|---|---|
|0|Model 0\(Baseline)|10% of data|None|EfficientNETB0|
|1|Model 1|1% of the data|Random flip,Rotation,Zoom,Height,Width,Data Augmentation|Same as model 0|
|2|Model 2|Same as model 0|Same as model 1|Same as model 0|
|3|Model 3|Same as model 0|Same as model 1|Fine tuning model trained on EfficientNETB0 with top 10 layers unfrozen|
|4|Model 4|100% of the training data|Same as model 1|Same as Model 3|


"""

# Sequential API of keras

"""
            # Architecture of the Sequential API
            
    sequential_model =  tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
        ], name="sequential_model")
    
    sequential_model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer = Adam(),
        metrics=["accuracy"]
        )
    
    sequential_model.fit(x_train,y_train,epochs=5,
                         batch_size=32)

"""


# Functional API of keras

"""
                # Architecture of the functional API
                
    inputs = tf.keras.layers.Input(shape=(28,28))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    
    func_model = tf.keras.Model(inputs, outputs, name="functional_model")
    
    func_model.compile(loss="categorical_crossentropy",
                       optimizer=Adam(),
                       metrics=["accuracy"])
    
    func_model.fit(x_train, y_train,
                   batch_size=32,
                   epochs=5)


"""

# Functional API is more flexible and is used to build more complex models


# BASELINE MODEL MODEL 0
base_model = EfficientNetB0(include_top=False)
base_model.trainable = False # freezing the base model
inputs = tf.keras.layers.Input(shape=(224,224,3),name="input_layer")

# if using a resnet50V2 then we need to normalize the inputs
x =  Rescaling(1./255)(inputs)
 

x =  base_model(inputs)
print(f"Shape after passing inputs through the base model : {x.shape}")
# Shape after passing inputs through the base model : (None, 7, 7, 1280)


# TODO -- Aggregrate all the info
x = GlobalAveragePooling2D(name = "global_avg_pooling_layer")(x)

print(f"Shape after passing inputs through global average pooling layer in the base model : {x.shape}")
# Shape after passing inputs through global average pooling layer in the base model : (None, 1280)

outputs = Dense(10, activation=softmax,name="output_layer")(x)

model_0 = Model(inputs,outputs)

model_0.compile(loss="categorical_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

base_model_history = model_0.fit(train_dt_10,
            steps_per_epoch=len(train_dt_10),
            validation_data=test_dt_10,
            validation_steps=len(test_dt_10),
            epochs=5,
            callbacks=[create_tensorboard_callback("tfhub", "EfficientNetB0-Feature Vector Base Model ")])

# loss_plotter(base_model_history)

model_0.evaluate(test_dt_10)

# check the layers in our model
for layer_num , layer in enumerate(base_model.layers):
    print(layer_num, layer.name)
    
"""
0 input_10
1 rescaling_18
2 normalization_9
3 tf.math.truediv_9
4 stem_conv_pad
5 stem_conv
6 stem_bn
7 stem_activation
8 block1a_dwconv
9 block1a_bn
10 block1a_activation
11 block1a_se_squeeze
12 block1a_se_reshape
13 block1a_se_reduce
14 block1a_se_expand
15 block1a_se_excite
16 block1a_project_conv
17 block1a_project_bn
18 block2a_expand_conv
19 block2a_expand_bn
20 block2a_expand_activation
21 block2a_dwconv_pad
22 block2a_dwconv
23 block2a_bn
24 block2a_activation
25 block2a_se_squeeze
26 block2a_se_reshape
27 block2a_se_reduce
28 block2a_se_expand
29 block2a_se_excite
30 block2a_project_conv
31 block2a_project_bn
32 block2b_expand_conv
33 block2b_expand_bn
34 block2b_expand_activation
35 block2b_dwconv
36 block2b_bn
37 block2b_activation
38 block2b_se_squeeze
39 block2b_se_reshape
40 block2b_se_reduce
41 block2b_se_expand
42 block2b_se_excite
43 block2b_project_conv
44 block2b_project_bn
45 block2b_drop
46 block2b_add
47 block3a_expand_conv
48 block3a_expand_bn
49 block3a_expand_activation
50 block3a_dwconv_pad
51 block3a_dwconv
52 block3a_bn
53 block3a_activation
54 block3a_se_squeeze
55 block3a_se_reshape
56 block3a_se_reduce
57 block3a_se_expand
58 block3a_se_excite
59 block3a_project_conv
60 block3a_project_bn
61 block3b_expand_conv
62 block3b_expand_bn
63 block3b_expand_activation
64 block3b_dwconv
65 block3b_bn
66 block3b_activation
67 block3b_se_squeeze
68 block3b_se_reshape
69 block3b_se_reduce
70 block3b_se_expand
71 block3b_se_excite
72 block3b_project_conv
73 block3b_project_bn
74 block3b_drop
75 block3b_add
76 block4a_expand_conv
77 block4a_expand_bn
78 block4a_expand_activation
79 block4a_dwconv_pad
80 block4a_dwconv
81 block4a_bn
82 block4a_activation
83 block4a_se_squeeze
84 block4a_se_reshape
85 block4a_se_reduce
86 block4a_se_expand
87 block4a_se_excite
88 block4a_project_conv
89 block4a_project_bn
90 block4b_expand_conv
91 block4b_expand_bn
92 block4b_expand_activation
93 block4b_dwconv
94 block4b_bn
95 block4b_activation
96 block4b_se_squeeze
97 block4b_se_reshape
98 block4b_se_reduce
99 block4b_se_expand
100 block4b_se_excite
101 block4b_project_conv
102 block4b_project_bn
103 block4b_drop
104 block4b_add
105 block4c_expand_conv
106 block4c_expand_bn
107 block4c_expand_activation
108 block4c_dwconv
109 block4c_bn
110 block4c_activation
111 block4c_se_squeeze
112 block4c_se_reshape
113 block4c_se_reduce
114 block4c_se_expand
115 block4c_se_excite
116 block4c_project_conv
117 block4c_project_bn
118 block4c_drop
119 block4c_add
120 block5a_expand_conv
121 block5a_expand_bn
122 block5a_expand_activation
123 block5a_dwconv
124 block5a_bn
125 block5a_activation
126 block5a_se_squeeze
127 block5a_se_reshape
128 block5a_se_reduce
129 block5a_se_expand
130 block5a_se_excite
131 block5a_project_conv
132 block5a_project_bn
133 block5b_expand_conv
134 block5b_expand_bn
135 block5b_expand_activation
136 block5b_dwconv
137 block5b_bn
138 block5b_activation
139 block5b_se_squeeze
140 block5b_se_reshape
141 block5b_se_reduce
142 block5b_se_expand
143 block5b_se_excite
144 block5b_project_conv
145 block5b_project_bn
146 block5b_drop
147 block5b_add
148 block5c_expand_conv
149 block5c_expand_bn
150 block5c_expand_activation
151 block5c_dwconv
152 block5c_bn
153 block5c_activation
154 block5c_se_squeeze
155 block5c_se_reshape
156 block5c_se_reduce
157 block5c_se_expand
158 block5c_se_excite
159 block5c_project_conv
160 block5c_project_bn
161 block5c_drop
162 block5c_add
163 block6a_expand_conv
164 block6a_expand_bn
165 block6a_expand_activation
166 block6a_dwconv_pad
167 block6a_dwconv
168 block6a_bn
169 block6a_activation
170 block6a_se_squeeze
171 block6a_se_reshape
172 block6a_se_reduce
173 block6a_se_expand
174 block6a_se_excite
175 block6a_project_conv
176 block6a_project_bn
177 block6b_expand_conv
178 block6b_expand_bn
179 block6b_expand_activation
180 block6b_dwconv
181 block6b_bn
182 block6b_activation
183 block6b_se_squeeze
184 block6b_se_reshape
185 block6b_se_reduce
186 block6b_se_expand
187 block6b_se_excite
188 block6b_project_conv
189 block6b_project_bn
190 block6b_drop
191 block6b_add
192 block6c_expand_conv
193 block6c_expand_bn
194 block6c_expand_activation
195 block6c_dwconv
196 block6c_bn
197 block6c_activation
198 block6c_se_squeeze
199 block6c_se_reshape
200 block6c_se_reduce
201 block6c_se_expand
202 block6c_se_excite
203 block6c_project_conv
204 block6c_project_bn
205 block6c_drop
206 block6c_add
207 block6d_expand_conv
208 block6d_expand_bn
209 block6d_expand_activation
210 block6d_dwconv
211 block6d_bn
212 block6d_activation
213 block6d_se_squeeze
214 block6d_se_reshape
215 block6d_se_reduce
216 block6d_se_expand
217 block6d_se_excite
218 block6d_project_conv
219 block6d_project_bn
220 block6d_drop
221 block6d_add
222 block7a_expand_conv
223 block7a_expand_bn
224 block7a_expand_activation
225 block7a_dwconv
226 block7a_bn
227 block7a_activation
228 block7a_se_squeeze
229 block7a_se_reshape
230 block7a_se_reduce
231 block7a_se_expand
232 block7a_se_excite
233 block7a_project_conv
234 block7a_project_bn
235 top_conv
236 top_bn
237 top_activation
"""

base_model.summary() # efficient net model summary
model_0.summary()

# __________________________________________________________________________________________________
# Model: "model_0"
# _________________________________________________________________
#   Layer (type)                Output Shape              Param #   
# =================================================================
#   input_layer (InputLayer)    [(None, 224, 224, 3)]     0         
                                                                 
#   efficientnetb0 (Functional)  (None, None, None, 1280)  4049571  
                                                                 
#   global_avg_pooling_layer (G  (None, 1280)             0         
#   lobalAveragePooling2D)                                          
                                                                 
#   output_layer (Dense)        (None, 10)                12810     
                                                                 
# =================================================================
# Total params: 4,062,381
# Trainable params: 12,810
# Non-trainable params: 4,049,571
# _________________________________________________________________

# Taking out the feature vector from the Model 0

"""
Notice the shape after passing the inputs in the model 
it is (None, 7, 7, 1280) and then after passing through the global
average max pooling 2D we get only 1280 and all the 1D dimensions get removed.
That global average max pooling 2D is the feature vector

"""
# Lets pass a random input tensor and check what happens as it passes
# through the global average max pooling 2D

in_shape = [1,2,3,4]

tf.random.set_seed(42)
input_tensor =  tf.random.normal(in_shape)
print(f"Randomised tensor : \n {input_tensor} \n")

# Pass it into the global average pooling layers

global_avg_pooled_tensor =  GlobalAveragePooling2D()(input_tensor)
print(f"Global average pooled tensor : \n {global_avg_pooled_tensor} \n")

print(f"The shape of the input_tenor : {input_tensor.shape}")
print(f"The shape of the global avergae max pooled tensor : {global_avg_pooled_tensor.shape}")

# Replication of the global avergae max pooling 2D 
# Just removing the even axis
# reducing the axis 1, 2 these are the even axis 
tf.reduce_mean(input_tensor, axis=(1,2)) # basically this is what the model 
# has learnt from the images tha we have passed

"""
Feature vector = A  feature vector is a learned representation of the input data(
    a compressed form of the input data based on how the model see's it.')

"""

###################################################################################

# MODEL 1

"""
Just using 1% of the data 
"""


# walker_dt_dir("10_food_classes_1_percent")

train_1_path = "10_food_classes_1_percent/train"
test_1_path = "10_food_classes_1_percent/test"


train_data_1p = image_dataset_from_directory(train_1_path,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode="categorical")

test_data_1p =  image_dataset_from_directory(test_1_path,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode="categorical")

# Augmentation on the 1 % dataset
# We will not augment on the fly we will augment the dataset directly on the 1% dataset
# When passed on as a layer to a model data augmentation is automatically turned on
# during the training but turned off during the testing data or unseen data

data_augmentation = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    # preprocessing.Rescale(1./255)
    ], name="augmented_layer")

# Lets view the augmented layers
target_class = random.choice(train_data_1p.class_names)
target_path = "10_food_classes_1_percent/train/"
target_dir = target_path + target_class
random_image = random.choice(os.listdir(target_dir))
random_image_path = target_dir + "/"  + random_image

# getting the original image
img = mpimg.imread(random_image_path)
plt.imshow(img)
plt.title(f"Original random image from class: {target_class}")
plt.axis(False)

# augment the image
aug_image = data_augmentation(img, training=True)
plt.figure()
plt.imshow(aug_image/255.)
plt.title(f"Augmented  random image from class : {target_class}")
plt.axis(False)

