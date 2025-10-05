import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, BatchNormalization,Layer, Multiply, Reshape, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K

K.set_image_data_format('channels_last')

# ---------------------- Attention 模块 ----------------------
class FlowAttention(Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.dim_reduce = max(4, channels//reduction)  # 确保最小维度
        
    def build(self, input_shape):
        self.W_squeeze = self.add_weight(
            name='W_squeeze', shape=(self.channels, self.dim_reduce),
            initializer='he_normal'
        )
        self.W_excite = self.add_weight(
            name='W_excite', shape=(self.dim_reduce, self.channels),
            initializer='he_normal'
        )
        self.conv_spatial = Conv1D(
            1, 3, padding='same', activation='sigmoid',
            kernel_initializer='he_normal'
        )
        super().build(input_shape)
        
    def call(self, x):
        # 通道注意力
        squeeze = K.mean(x, axis=1)
        excitation = K.dot(squeeze, self.W_squeeze)
        excitation = K.relu(excitation)
        excitation = K.dot(excitation, self.W_excite)
        excitation = K.sigmoid(excitation)
        excitation = Reshape((1, self.channels))(excitation)
        x = Multiply()([x, excitation])
        
        # 空间注意力
        spatial_att = self.conv_spatial(x)
        return Multiply()([x, spatial_att])
    
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = super().get_config()
        config.update({
        'channels': self.channels,
        'reduction': self.reduction
        })
        return config

ss = 10

def crossentropy_cut(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask=K.greater_equal(y_true_f,-0.5)
    losses = -(y_true_f * K.log(y_pred_f) + (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    losses = tf.boolean_mask(losses, mask)
    masked_loss = tf.reduce_mean(losses)
    return masked_loss

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters, (kernel_size,1), strides=(strides,1), padding=padding)(x)
    return Lambda(lambda x: K.squeeze(x, axis=2))(x)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5), dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + ss) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + ss)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# ---------------------- pcc/ucc函数 ----------------------
def pcc(layer_in, num_filter, size_kernel, activation='relu', padding='same'):
    x = MaxPooling1D(pool_size=2)(layer_in)
    x = BatchNormalization()(Conv1D(num_filter, size_kernel, activation=activation, padding=padding)(x))
    x = BatchNormalization()(Conv1D(num_filter, size_kernel, activation=activation, padding=padding)(x))
    x = FlowAttention(num_filter)(x)  
    return x

def ucc(layer_in1,layer_in2, num_filter, size_kernel, activation='relu', padding='same'):
    x = concatenate([Conv1DTranspose(layer_in1,num_filter,2,strides=2,padding=padding), layer_in2], axis=2)
    x = BatchNormalization()(Conv1D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    x = BatchNormalization()(Conv1D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    x = FlowAttention(num_filter)(x)  
    return x

# ---------------------- get_unet函数 ----------------------
def get_unet(the_lr=1e-3, num_class=1, num_channel=6, size=10240):  # 
    inputs = Input((size, num_channel))
    
    num_blocks=5 
    initial_filter=32  
    scale_filter=1.5
    size_kernel=7
    activation='relu'
    padding='same'    

    layer_down=[]
    layer_up=[]

    conv0 = BatchNormalization()(Conv1D(initial_filter, size_kernel, activation=activation, padding=padding)(inputs))
    conv0 = BatchNormalization()(Conv1D(initial_filter, size_kernel, activation=activation, padding=padding)(conv0))
    layer_down.append(conv0)
    num=initial_filter

    for i in range(num_blocks):
        num=int(num * scale_filter)
        the_layer=pcc(layer_down[i], num, size_kernel)
        layer_down.append(the_layer)

    layer_up.append(the_layer)
    for i in range(num_blocks):
        num=int(num / scale_filter)
        the_layer=ucc(layer_up[i],layer_down[-(i+2)],num, size_kernel)
        layer_up.append(the_layer)

    convn = Conv1D(num_class, 1, activation='sigmoid', padding=padding)(layer_up[-1])

    model = Model(inputs=[inputs], outputs=[convn])
    model.compile(optimizer=Adam(lr=the_lr), loss=crossentropy_cut, metrics=[dice_coef])
    return model