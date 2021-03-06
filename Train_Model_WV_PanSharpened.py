import tensorflow as tf
import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, osr
import pandas as pd
import matplotlib.pyplot as plt
import glob
import datetime

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
    def __init__(self,image_filenames,labels,batch_size,input_shape):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self,idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        new_x,new_y = load_images(batch_x,batch_y,self.input_shape)
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

def load_images(train_list,label_list,input_shape):
    train_stack = np.empty((len(train_list),input_shape[0],input_shape[1],input_shape[2]),dtype=np.float16)
    label_stack = np.empty((len(label_list),input_shape[0],input_shape[1],1),dtype=np.uint8)
    for i in range(len(train_list)):
        train_image = train_list[i]
        label_image = label_list[i]
        src_train = gdal.Open(train_image,gdalconst.GA_ReadOnly)
        red_channel = np.array(src_train.GetRasterBand(1).ReadAsArray())
        green_channel = np.array(src_train.GetRasterBand(2).ReadAsArray())
        blue_channel = np.array(src_train.GetRasterBand(3).ReadAsArray())
        red_channel = red_channel.astype(float)/2047 #11 bit -> [0,2047]
        green_channel = green_channel.astype(float)/2047 #11 bit -> [0,2047]
        blue_channel = blue_channel.astype(float)/2047 #11 bit -> [0,2047]
        rgb_channel = np.dstack((red_channel,green_channel,blue_channel))
        train_stack[i] = rgb_channel
        src_label = gdal.Open(label_image,gdalconst.GA_ReadOnly)
        label_channel = np.array(src_label.GetRasterBand(1).ReadAsArray())
        label_stack[i,:,:,0] = label_channel
        train_array = tf.convert_to_tensor(train_stack,dtype=tf.float16)
        label_array = tf.convert_to_tensor(label_stack,dtype=tf.float16)
        label_array = tf.expand_dims(label_array,axis=-1)
    return train_array,label_array

'''
def load_data(main_dir):
    training_data_dir = f'{main_dir}Training_Data/subimages/'
    labels_dir = f'{main_dir}Labels/subimages/'
    training_data = glob.glob(f'{training_data_dir}*.tif')
    training_data.sort()
    label_data = glob.glob(f'{labels_dir}*.tif')
    label_data.sort()

    n_files = len(training_data)
    n_val = int(np.round(0.2*n_files))
    n_test = int(np.round(0.2*n_files))
    n_train = int(n_files - n_val - n_test)

    idx_shuffle = np.arange(n_files)
    np.random.shuffle(idx_shuffle)
    idx_train = idx_shuffle[:n_train]
    idx_val = idx_shuffle[n_train:n_train+n_val]
    idx_test = idx_shuffle[n_train+n_val:]

    train_list = np.array(training_data)[idx_train.astype(int)]
    val_list = np.array(training_data)[idx_val.astype(int)]
    test_list = np.array(training_data)[idx_test.astype(int)]
    train_label_list = np.array(label_data)[idx_train.astype(int)]
    val_label_list = np.array(label_data)[idx_val.astype(int)]
    test_label_list = np.array(label_data)[idx_test.astype(int)]

    return train_list,train_label_list,val_list,val_label_list,test_list,test_label_list
'''

def load_data(main_dir,N_PATCHES):
    training_data_dir = f'{main_dir}Training_Data/subimages/'
    labels_dir = f'{main_dir}Labels/subimages/'
    unique_locs = sorted(glob.glob(f'{training_data_dir}*000001.tif'))
    n_locs = len(unique_locs)
    n_val = int(np.round(0.2*n_locs))
    n_test = int(np.round(0.2*n_locs))
    n_train = int(n_locs - n_val - n_test)
    
    idx_shuffle = np.arange(n_locs)
    np.random.shuffle(idx_shuffle)
    idx_train = idx_shuffle[:n_train]
    idx_val = idx_shuffle[n_train:n_train+n_val]
    idx_test = idx_shuffle[n_train+n_val:]

    train_locs = np.array(unique_locs)[idx_train.astype(int)]
    val_locs = np.array(unique_locs)[idx_val.astype(int)]
    test_locs = np.array(unique_locs)[idx_test.astype(int)]

    print('Training Locations:')
    [print(loc.split('/')[-1].split('_WV')[0]) for loc in np.sort(train_locs)]
    print('')
    print('Validation Locations:')
    [print(loc.split('/')[-1].split('_WV')[0]) for loc in np.sort(val_locs)]
    print('')
    print('Test Locations:')
    [print(loc.split('/')[-1].split('_WV')[0]) for loc in np.sort(test_locs)]
    print('')

    train_list = np.empty([0,1],dtype=str)
    val_list = np.empty([0,1],dtype=str)
    test_list = np.empty([0,1],dtype=str)
    train_label_list = np.empty([0,1],dtype=str)
    val_label_list = np.empty([0,1],dtype=str)
    test_label_list = np.empty([0,1],dtype=str)

    for loc in train_locs:
        tmp_train_list,tmp_train_label_list = get_images(loc,N_PATCHES)
        train_list = np.append(train_list,tmp_train_list)
        train_label_list = np.append(train_label_list,tmp_train_label_list)
    for loc in val_locs:
        tmp_val_list,tmp_val_label_list = get_images(loc,N_PATCHES)
        val_list = np.append(val_list,tmp_val_list)
        val_label_list = np.append(val_label_list,tmp_val_label_list)
    for loc in test_locs:
        tmp_test_list,tmp_test_label_list = get_images(loc,N_PATCHES)
        test_list = np.append(test_list,tmp_test_list)
        test_label_list = np.append(test_label_list,tmp_test_label_list)
    return train_list,train_label_list,val_list,val_label_list,test_list,test_label_list

def get_images(loc,N_PATCHES):
    n_images = len(glob.glob(loc.replace('000001.tif','*.tif')))
    idx_loc = np.sort(np.random.choice(n_images,N_PATCHES,replace=False))
    tmp_list = np.asarray([loc.replace('_000001.tif',f'_{i:06d}.tif') for i in idx_loc])
    tmp_label_list = np.asarray([t.replace('Training_Data','Labels').replace('_pansharpened_orthorectified_','_label_') for t in tmp_list])
    return tmp_list,tmp_label_list




def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[2:4],'GPU')

    LEARNING_RATE = 0.001 #Default for TF is 0.001
    EPSILON = 1e-7 #Default is 1e-7
    BATCH_SIZE = 20
    EPOCHS = 50
    INPUT_SHAPE = (224,224,3)
    N_PATCHES = 2000
    SEED = 16
    PATIENCE = 10

    main_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/Machine_Learning/WV_PanSharpened/'
    models_dir = f'{main_dir}Models/'

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,epsilon=EPSILON)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    np.random.seed(seed=SEED)

    #TO DO:
    '''
    clip out data frames that are all 0 (i.e. no data)
    split data into train/validation/test, assign to files, not arrays
        e.g. Djibouti is validation, NY in train, etc
        need a lot more training data
    select loss: binary crossentropy or sparse categorical crossentropy
    '''

    train_list,train_label_list,val_list,val_label_list,test_list,test_label_list = load_data(main_dir,N_PATCHES)
    training_batch_generator = Custom_Generator(train_list, train_label_list, BATCH_SIZE,INPUT_SHAPE)
    validation_batch_generator = Custom_Generator(val_list, val_label_list, BATCH_SIZE,INPUT_SHAPE)

    now = datetime.datetime.now()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=PATIENCE),
        tf.keras.callbacks.CSVLogger(f'{models_dir}Train_Val_Accuracy_Epochs_{now.strftime("%Y%m%d")}.txt', separator=',', append=False),
        tf.keras.callbacks.ModelCheckpoint(f'{models_dir}Model_{now.strftime("%Y%m%d")}.h5', monitor='val_loss', save_best_only=False)
        ]

    model = build_resunet_model(INPUT_SHAPE)
    # model.summary()
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    model.fit(training_batch_generator,
        steps_per_epoch = int(len(train_list) / BATCH_SIZE),
        epochs = EPOCHS,
        verbose = 1,
        validation_data = validation_batch_generator,
        validation_steps = int(len(val_list) / BATCH_SIZE),
        callbacks=callbacks
    )
    model.save(f'{models_dir}resunet_model_wv_pansharpened_{now.strftime("%Y%m%d")}')


if __name__ == '__main__':
    main()