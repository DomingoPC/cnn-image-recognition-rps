# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:39:05 2024

@author: domin
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten, Input, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# --- Modelo 1 ---
def modelo1(image_size=(128,128)):
    input_layer = Input(shape=(*(image_size), 3))
    
    # Initial Conv + Pooling
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    
    # Example of Conv Blocks
    def conv_block(x, filters):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x
    
    # Block 1
    x1 = conv_block(x, 64)
    # x1 = MaxPooling2D((2, 2))(x1)
    # x1 = Conv2D(128, (1, 1), activation='relu')(x1)
    
    # Block 2
    x2 = conv_block(x, 128)
    # x2 = MaxPooling2D((2, 2))(x2)
    
    # Concatenate
    concatenated = Concatenate()([x1, x2])
    concatenated = MaxPooling2D((2, 2))(concatenated)
    
    # Final Convolution
    x = Conv2D(256, (3, 3), activation='relu')(concatenated)
    x = MaxPooling2D((2, 2))(x)
    
    # Fully Connected
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(3, activation='softmax')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)



# --- Modelo 2 ---
def modelo2(image_size=(128,128)):
    def inception(kernels, x):
        xa = Conv2D(kernels, (1,1), padding='same')(x)
        xa = Conv2D(kernels*2, (3,3), padding='same')(xa)
    
        xb = Conv2D(kernels, (1,1), padding='same')(x)
        xb = Conv2D(kernels*2, (5,5), padding='same')(xb)
    
        xc = Conv2D(kernels, (1,1), padding='same')(x)
        xc = Conv2D(kernels*2, (7,7), padding='same')(xc)
        
        out = Concatenate(axis=-1)([xa, xb, xc])
        return out
    
    
    def bloque_residual(kernels, x):
        inception_path = inception(kernels, x)
        inception_path = Dropout(0.4)(inception_path)
        # inception_path = MaxPooling2D((2,2), strides=(2,2), padding='same')(inception_path)
        # inception_path = inception(kernels, inception_path)
        # inception_path = MaxPooling2D((2,2), strides=(2,2), padding='same')(inception_path)
        inception_path = Conv2D(kernels, (1,1), padding='same')(inception_path)
        
        out = Add()([x, inception_path])
        out = tf.keras.layers.Activation('relu')(out)
        return out
    
    
    entrada = Input(shape=(*image_size, 3))
    
    # Hemos visto de la vgg que la primera convolucional saca bastante información
    # sobre las manos, así que empezamos con una similar
    x = Conv2D(64, (3,3), padding='same', activation='relu', name='Conv2d_1')(entrada)
    x = Dropout(0.4)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = bloque_residual(64, x)
    
    # Reducción de filtros de convolución
    x = Conv2D(128, (3,3), activation='relu', name='conv2d_reduccion')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((3,3), strides=(3,3), padding='same')(x)
    
    # Toma de decisiones
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    salida = Dense(3, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs=entrada, outputs=salida)



# --- Modelo 3 : modelo propio sin regularización ---
# https://stackoverflow.com/questions/50940827/should-you-always-use-regularization-tensorflow
# https://datascience.stackexchange.com/questions/13031/should-i-use-regularization-every-time

def modelo_sinRegularizacion(image_size=(128,128)):
    entrada = Input(shape=(*image_size, 3))
    
    x = Conv2D(64, (3,3), padding='same', activation='relu', name='conv2d_entrada')(entrada)
    x = Dropout(0.4)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = Conv2D(128, (5,5), activation='relu',
               name='conv2d_2_5x5')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    
    x = Conv2D(256, (3,3), activation='relu',
               name='conv2d_3_3x3')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    
    # Bloque de dos convolucionales
    z1 = Conv2D(512, (7,7), activation='relu', padding='same',
                name='conv2d_bloque_7x7')(x)
    z1 = Dropout(0.4)(z1)
    
    z2 = Conv2D(512, (3,3), activation='relu', padding='same',
                name='conv2d_bloque_3x3')(x)
    z2 = Dropout(0.5)(z2)
    
    x = Concatenate()([z1,z2])
    x = MaxPooling2D((3,3), strides=(3,3), padding='same')(x)
    
    # Reducción de número de filtros
    x = Conv2D(512, (1,1), activation='relu', padding='same', 
               name='conv2d_reduce')(x)
    
    # Toma de decisiones
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    salida = Dense(3, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs=entrada, outputs=salida)


# --- Modelo 4 : modelo propio con regularización---
# https://stackoverflow.com/questions/50940827/should-you-always-use-regularization-tensorflow
# https://datascience.stackexchange.com/questions/13031/should-i-use-regularization-every-time

def modelo_conRegularizacion(image_size=(128,128)):
    entrada = Input(shape=(*image_size, 3))
    
    x = Conv2D(64, (3,3), padding='same', activation='relu', 
               name='conv2d_entrada',
               kernel_regularizer=l2(1e-3))(entrada)
    x = Dropout(0.4)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = Conv2D(128, (5,5), activation='relu',
               name='conv2d_2_5x5')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    
    x = Conv2D(256, (3,3), activation='relu',
               name='conv2d_3_3x3')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    
    # Bloque de dos convolucionales
    z1 = Conv2D(512, (7,7), activation='relu', padding='same',
                name='conv2d_bloque_7x7')(x)
    z1 = Dropout(0.4)(z1)
    
    z2 = Conv2D(512, (3,3), activation='relu', padding='same',
                name='conv2d_bloque_3x3')(x)
    z2 = Dropout(0.5)(z2)
    
    x = Concatenate()([z1,z2])
    x = MaxPooling2D((3,3), strides=(3,3), padding='same')(x)
    
    # Reducción de número de filtros
    x = Conv2D(512, (1,1), activation='relu', padding='same', 
               name='conv2d_reduce')(x)
    
    # Toma de decisiones
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    salida = Dense(3, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs=entrada, outputs=salida)


# --- Modelo VGG16 ---
# https://keras.io/api/applications/
def modeloVGG16(image_size=(128,128)):
    # Modelo VGG16
    parent_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(*image_size, 3))
    
    parent_model.trainable = False
    
    # Construímos el modelo
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(*image_size, 3)))
    model.add(parent_model)
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    
    model.add(Dense(3, activation='softmax'))
    
    return model


# --- Modelo ResNet ---
# https://keras.io/api/applications/
def modeloResNet(image_size=(128,128)):
    # Modelo ResNet50
    parent_model = tf.keras.applications.ResNet101V2(
        weights='imagenet',
        include_top=False,
        input_shape=(*image_size, 3))
    
    parent_model.trainable = False
    
    # Construímos el modelo
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(*image_size, 3)))
    model.add(parent_model)
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    
    model.add(Dense(3, activation='softmax'))
    
    return model