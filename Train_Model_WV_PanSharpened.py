import tensorflow as tf
import numpy as np
# import geopandas as gpd




def build_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    #ENCODING
    x = conv_BN_activation_block(inputs, 64, (3,3), strides=1, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(64, (1,1), strides=1, padding='same')(inputs)
    s1 = tf.keras.layers.Add()([x, s])

    x = BN_activation_block(s1, 'relu')
    x = conv_BN_activation_block(x, 128, (3,3), strides=2, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(128, (1,1), strides=2, padding='same')(x)
    s2 = tf.keras.layers.Add()([x, s])

    x = BN_activation_block(s2, 'relu')
    x = conv_BN_activation_block(x, 256, (3,3), strides=2, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(256, (1,1), strides=2, padding='same')(x)
    s3 = tf.keras.layers.Add()([x, s])

    #BRIDGE
    x = BN_activation_block(s3, 'relu')
    x = conv_BN_activation_block(x, 512, (3,3), strides=2, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(512, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(512, (1,1), strides=2, padding='same')(x)
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
    s = tf.keras.layers.Conv2D(128, (1,1), strides=1, padding='same')(d3)
    x = tf.keras.layers.Add()([x, s])

    x = tf.keras.layers.UpSampling2D((2,2))(x)
    d1 = tf.keras.layers.Concatenate()([x, s1])
    x = BN_activation_block(d1, 'relu')
    x = conv_BN_activation_block(x, 64, (3,3), strides=1, padding='same', activation='relu')
    x = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same')(x)
    s = tf.keras.layers.Conv2D(64, (1,1), strides=1, padding='same')(d3)
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

def main():
    input_shape = (224,224,3)
    model = build_model(input_shape)
    model.summary()

if __name__ == '__main__':
    main()