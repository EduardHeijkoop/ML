from re import X
import tensorflow as tf
import numpy as np

input_shape = (224,224,3)
inputs = tf.keras.layers.Input(input_shape)

# 
x = tf.keras.layers.Conv2D(64, (3,3),strides=1,padding='same') (inputs)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(64, (3,3),strides=1,padding='same') (x)
s = tf.keras.layers.Conv2D(64, (1,1),strides=1,padding='same') (inputs)
s1 = x + s 

x = tf.keras.layers.BatchNormalization() (s1)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(128, (3,3),strides=2,padding='same') (x)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(128, (3,3),strides=1,padding='same') (x)
s = tf.keras.layers.Conv2D(128, (1,1),strides=2,padding='same') (s1)
s2 = x + s

x = tf.keras.layers.BatchNormalization() (s2)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(256, (3,3),strides=2,padding='same') (x)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(256, (3,3),strides=1,padding='same') (x)
s = tf.keras.layers.Conv2D(256, (1,1),strides=2,padding='same') (s2)
s3 = x + s

x = tf.keras.layers.BatchNormalization() (s3)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(512, (3,3),strides=2,padding='same') (x)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(512, (3,3),strides=1,padding='same') (x)
s = tf.keras.layers.Conv2D(512, (1,1),strides=2,padding='same') (s3)
b = x + s

x = tf.keras.layers.UpSampling2D((2,2)) (b)
d3 = tf.keras.layers.Concatenate() ([x,s3])
x = tf.keras.layers.BatchNormalization() (d3)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(256, (3,3),strides=1,padding='same') (x)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(256, (3,3),strides=1,padding='same') (x)
s = tf.keras.layers.Conv2D(256, (1,1),strides=1,padding='same') (d3)
x = x + s

x = tf.keras.layers.UpSampling2D((2,2)) (x)
d2 = tf.keras.layers.Concatenate() ([x,s2])
x = tf.keras.layers.BatchNormalization() (d2)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(128, (3,3),strides=1,padding='same') (x)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(128, (3,3),strides=1,padding='same') (x)
s = tf.keras.layers.Conv2D(128, (1,1),strides=1,padding='same') (d2)
x = x + s

x = tf.keras.layers.UpSampling2D((2,2)) (x)
d1 = tf.keras.layers.Concatenate() ([x,s1])
x = tf.keras.layers.BatchNormalization() (d1)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(64, (3,3),strides=1,padding='same') (x)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation('relu') (x)
x = tf.keras.layers.Conv2D(64, (3,3),strides=1,padding='same') (x)
s = tf.keras.layers.Conv2D(64, (1,1),strides=1,padding='same') (d1)
x = x + s

x = tf.keras.layers.Conv2D(1, (1,1),strides=1,padding='same') (x)
outputs = tf.keras.layers.Activation('sigmoid') (x)

model = tf.keras.Model(inputs=inputs, outputs=outputs,name='ResUNet')
print('plotting model')
tf.keras.utils.plot_model(model,to_file='/home/heijkoop/Desktop/ResUNet/TF_ResUNet.png',show_shapes=True)
print('model plotted')
model.summary()
