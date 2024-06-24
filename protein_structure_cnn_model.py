from keras.layers import Conv3D, Dense, Flatten, MaxPooling3D,Conv3DTranspose, Input
from keras.models import Model
import numpy as np


def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(encoded)
    x = Conv3DTranspose(128, kernel_size=(3, 3, 3), activation='relu', padding='same', strides=(2, 2, 2))(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = Conv3DTranspose(64, kernel_size=(3, 3, 3), activation='relu', padding='same', strides=(2, 2, 2))(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = Conv3DTranspose(32, kernel_size=(3, 3, 3), activation='relu', padding='same', strides=(2, 2, 2))(x)
    decoded = Conv3D(input_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    return autoencoder, encoder


def build_simple_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(inputs)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Flatten()(x)
    features = Dense(512, activation='relu')(x)

    model = Model(inputs, features)
    return model