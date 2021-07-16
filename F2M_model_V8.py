# -*- coding:utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2(0.000002)

Transpose_conv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                                                  padding="same", use_bias=False, kernel_regularizer=l2)

Transpose_conv2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                                                  padding="same", use_bias=False, kernel_regularizer=l2)

Dense_1_1 = tf.keras.layers.Dense(64 // 2, kernel_regularizer=l2)
Dense_1_2 = tf.keras.layers.Dense(64, kernel_regularizer=l2)

Dense_2_1 = tf.keras.layers.Dense(32 // 2, kernel_regularizer=l2)
Dense_2_2 = tf.keras.layers.Dense(32, kernel_regularizer=l2)

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
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

def residual_block(x, kernel_size=3):
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=1, padding="same", use_bias=False, kernel_regularizer=l2)(x)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    return x + InstanceNormalization()(tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=1, padding="same", use_bias=False, kernel_regularizer=l2)(h))

def F2M_modelV8(input_shape=(256, 256, 3)):

    h = inputs = tf.keras.Input(input_shape)
    #######################################################################################################################################
    h = tf.keras.layers.ZeroPadding2D((4,4))(h)
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=9, strides=1, padding="valid", use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 32]
    EN_SE_1 = h

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 64]
    EN_SE_2 = h

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 128]

    for _ in range(8):
        h = residual_block(h, 3)
    #######################################################################################################################################

    h_1 = h
    h_2 = h
    h_3 = h

    # sharing the weight로 변경해보자
    #######################################################################################################################################
    h = Transpose_conv1(h_1)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 64]

    DE_SE_2 = tf.keras.layers.GlobalAveragePooling2D()(h)
    DE_SE_2 = Dense_1_1(DE_SE_2)
    DE_SE_2 = tf.keras.layers.ReLU()(DE_SE_2)
    DE_SE_2 = Dense_1_2(DE_SE_2)
    DE_SE_2 = tf.reshape(tf.nn.sigmoid(DE_SE_2), [-1, 1, 1, 64]) * h
    h = DE_SE_2 + EN_SE_2

    h = Transpose_conv2(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 32]

    DE_SE_1 = tf.keras.layers.GlobalAveragePooling2D()(h)
    DE_SE_1 = Dense_2_1(DE_SE_1)
    DE_SE_1 = tf.keras.layers.ReLU()(DE_SE_1)
    DE_SE_1 = Dense_2_2(DE_SE_1)
    DE_SE_1 = tf.reshape(tf.nn.sigmoid(DE_SE_1), [-1, 1, 1, 32]) * h
    h = DE_SE_1 + EN_SE_1

    h = tf.keras.layers.ZeroPadding2D((4,4))(h)
    h_1 = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="valid")(h)
    # https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py
    # https://github.com/lengstrom/fast-style-transfer/issues/79
    # https://github.com/lengstrom/fast-style-transfer/issues/79
    #######################################################################################################################################

    #######################################################################################################################################
    h = Transpose_conv1(h_2)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, EN_SE_2], -1)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same",
                               use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 64]
    add_h_1 = h

    DE_SE_2 = tf.keras.layers.GlobalAveragePooling2D()(h)
    DE_SE_2 = Dense_1_1(DE_SE_2)
    DE_SE_2 = tf.keras.layers.ReLU()(DE_SE_2)
    DE_SE_2 = Dense_1_2(DE_SE_2)
    DE_SE_2 = tf.reshape(tf.nn.sigmoid(DE_SE_2), [-1, 1, 1, 64]) * h
    h = DE_SE_2 + EN_SE_2

    h = Transpose_conv2(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, EN_SE_1], -1)
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same",
                               use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 32]
    add_h_2 = h

    DE_SE_1 = tf.keras.layers.GlobalAveragePooling2D()(h)
    DE_SE_1 = Dense_2_1(DE_SE_1)
    DE_SE_1 = tf.keras.layers.ReLU()(DE_SE_1)
    DE_SE_1 = Dense_2_2(DE_SE_1)
    DE_SE_1 = tf.reshape(tf.nn.sigmoid(DE_SE_1), [-1, 1, 1, 32]) * h
    h = DE_SE_1 + EN_SE_1

    h = tf.keras.layers.ZeroPadding2D((4,4))(h)
    h_2 = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="valid")(h)

    #######################################################################################################################################

    #######################################################################################################################################
    h = Transpose_conv1(h_3)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, EN_SE_2], -1)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same",
                               use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h + add_h_1)   # [128, 128, 64]

    DE_SE_2 = tf.keras.layers.GlobalAveragePooling2D()(h)
    DE_SE_2 = Dense_1_1(DE_SE_2)
    DE_SE_2 = tf.keras.layers.ReLU()(DE_SE_2)
    DE_SE_2 = Dense_1_2(DE_SE_2)
    DE_SE_2 = tf.reshape(tf.nn.sigmoid(DE_SE_2), [-1, 1, 1, 64]) * h
    h = DE_SE_2 + EN_SE_2

    h = Transpose_conv2(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, EN_SE_1], -1)
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same",
                               use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h + add_h_2)   # [256, 256, 32]

    DE_SE_1 = tf.keras.layers.GlobalAveragePooling2D()(h)
    DE_SE_1 = Dense_2_1(DE_SE_1)
    DE_SE_1 = tf.keras.layers.ReLU()(DE_SE_1)
    DE_SE_1 = Dense_2_2(DE_SE_1)
    DE_SE_1 = tf.reshape(tf.nn.sigmoid(DE_SE_1), [-1, 1, 1, 32]) * h
    h = DE_SE_1 + EN_SE_1

    h = tf.keras.layers.ZeroPadding2D((4,4))(h)
    h_3 = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="valid")(h)

    h = (tf.nn.softmax(h_1, -1) * h_2) + h_3    # 이 부분을 손봐야 할것같음!!!
    h = tf.nn.tanh(h)
    #######################################################################################################################################

    return tf.keras.Model(inputs=inputs, outputs=h)

def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)


    return tf.keras.Model(inputs=inputs, outputs=h)
