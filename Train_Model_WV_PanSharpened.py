import tensorflow as tf
import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, osr
import pandas as pd
import matplotlib.pyplot as plt

class InstanceNormalization(tf.keras.layers.Layer):
  #Instance Normalization Layer (https://arxiv.org/abs/1607.08022).
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class Custom_Generator(tf.keras.utils.Sequence):
    def __init__(self,image_filenames,labels,batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self,idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        new_x,new_y = load_images(batch_x,batch_y)
        # return batch_x,batch_y
        #edit this
        return new_x,new_y



def build_resunet_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    #ENCODING
    x = conv_BN_activation_block(inputs, 64, (3,3), strides=1, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(64, (1,1), strides=1, padding='same')(inputs)
    s1 = tf.keras.layers.Add()([x, s])

    x = BN_activation_block(s1, 'relu')
    x = conv_BN_activation_block(x, 128, (3,3), strides=2, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(128, (1,1), strides=2, padding='same')(s1)
    s2 = tf.keras.layers.Add()([x, s])

    x = BN_activation_block(s2, 'relu')
    x = conv_BN_activation_block(x, 256, (3,3), strides=2, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(256, (1,1), strides=2, padding='same')(s2)
    s3 = tf.keras.layers.Add()([x, s])

    #BRIDGE
    x = BN_activation_block(s3, 'relu')
    x = conv_BN_activation_block(x, 512, (3,3), strides=2, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(512, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(512, (1,1), strides=2, padding='same')(s3)
    b = tf.keras.layers.Add()([x, s])

    #DECODING
    x = tf.keras.layers.UpSampling2D((2,2))(b)
    d3 = tf.keras.layers.Concatenate()([x, s3])
    x = BN_activation_block(d3, 'relu')
    x = conv_BN_activation_block(x, 256, (3,3), strides=1, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(256, (1,1), strides=1, padding='same')(d3)
    x = tf.keras.layers.Add()([x, s])

    x = tf.keras.layers.UpSampling2D((2,2))(x)
    d2 = tf.keras.layers.Concatenate()([x, s2])
    x = BN_activation_block(d2, 'relu')
    x = conv_BN_activation_block(x, 128, (3,3), strides=1, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(128, (1,1), strides=1, padding='same')(d2)
    x = tf.keras.layers.Add()([x, s])

    x = tf.keras.layers.UpSampling2D((2,2))(x)
    d1 = tf.keras.layers.Concatenate()([x, s1])
    x = BN_activation_block(d1, 'relu')
    x = conv_BN_activation_block(x, 64, (3,3), strides=1, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(64, (1,1), strides=1, padding='same')(d1)
    x = tf.keras.layers.Add()([x, s])

    x = tf.keras.layers.Conv2D(1, (1,1),strides=1,padding='same') (x)
    outputs = tf.keras.layers.Activation('sigmoid') (x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name='ResUNet')

    return model

def conv_BN_activation_block(inputs, filters, kernel_size, strides, padding, activation):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(inputs)
    x = BN_activation_block(x, activation)
    return x

def BN_activation_block(inputs, activation):
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Activation(activation)(x)
    return x

def load_images(image_list,input_shape):
    for image in image_list:
        src_train = gdal.Open(image,gdalconst.GA_ReadOnly)
        red_channel = np.array(src_train.GetRasterBand(1).ReadAsArray())
        green_channel = np.array(src_train.GetRasterBand(2).ReadAsArray())
        blue_channel = np.array(src_train.GetRasterBand(3).ReadAsArray())
        red_channel = red_channel.astype(float)/2047
        green_channel = green_channel.astype(float)/2047
        blue_channel = blue_channel.astype(float)/2047




def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[3],'GPU')

    LEARNING_RATE = 0.001 #Default for TF is 0.001
    EPSILON = 1e-7 #Default is 1e-7
    BATCH_SIZE = 4
    main_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/Machine_Learning/WV_PanSharpened/'
    training_data_dir = f'{main_dir}Training_Data/subimages/'
    labels_dir = f'{main_dir}Labels/subimages/'
    models_dir = f'{main_dir}Models/'

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,epsilon=EPSILON)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    #TO DO:
    '''
    Load data
    normalize data (11 bit -> divide by 2047)
    clip out data frames that are all 0 (i.e. no data)
    split data into train/validation/test, assign to files, not arrays
        e.g. Djibouti is validation, NY in train, etc
        need a lot more training data
    create training and validation batch generators
    select loss: binary crossentropy or sparse categorical crossentropy

    '''



    input_shape = (224,224,3)
    model = build_resunet_model(input_shape)
    # model.summary()
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    model.fit(training_batch_generator,
        steps_per_epoch = int(len(train_list) / BATCH_SIZE),
        epochs = 100,
        verbose = 1,
        validation_data = validation_batch_generator,
        validation_steps = int(len(val_list) / BATCH_SIZE)
    )
    model.save('resunet_model_wv_pansharpened')


if __name__ == '__main__':
    main()