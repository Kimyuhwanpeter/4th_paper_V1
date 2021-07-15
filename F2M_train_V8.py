# -*- coding:utf-8 -*-
from F2M_model_V8 import *
from random import shuffle, random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256, 
                           
                           "load_size": 276,
                           
                           "batch_size": 1,
                           
                           "epochs": 200,
                           
                           "lr": 0.0002,
                           
                           "A_txt_path": "D:/[1]DB/[5]4th_paper_DB/female_train.txt",
                           
                           "A_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/AFAD/All/female_40_63/",
                           
                           "B_txt_path": "D:/[1]DB/[5]4th_paper_DB/male_train.txt",
                           
                           "B_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/AFAD/All/male_40_63/",

                           "age_range": [40, 64],

                           "n_classes": 24,

                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": ""})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def input_func(A_data, B_data):

    A_img = tf.io.read_file(A_data[0])
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3])
    A_img = A_img / 127.5 - 1.

    B_img = tf.io.read_file(B_data[0])
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.load_size, FLAGS.load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.img_size, FLAGS.img_size, 3])
    B_img = B_img / 127.5 - 1.

    A_lab = A_data[1] - FLAGS.age_range[0]
    B_lab = B_data[1] - FLAGS.age_range[0]

    return A_img, A_lab, B_img, B_lab

#@tf.function
def model_out(model, images, training=True):
    return model(images, training=training)

def cal_loss(A2B_G_model, B2A_G_model, A_discriminator, B_discriminator,
             A_batch_images, A_batch_labels, B_batch_images, B_batch_labels,
             extract_feature_model):

    with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape() as d_tape:
        fake_B = model_out(A2B_G_model, A_batch_images, True)
        fake_A_ = model_out(B2A_G_model, fake_B, True)
        
        fake_A = model_out(B2A_G_model, B_batch_images, True)
        fake_B_ = model_out(A2B_G_model, fake_A, True)

        DA_real = model_out(A_discriminator, A_batch_images, True)
        DA_fake = model_out(A_discriminator, fake_A, True)
        DB_real = model_out(B_discriminator, B_batch_images, True)
        DB_fake = model_out(B_discriminator, fake_B, True)

        ################################################################################################
        # 나이에 대한 distance를 구하는곳
        # feature vector로 뽑아야하기 때문에 어떻게 뽑아야할지가 강권 (pre-train model을 사용할까?)
        vector_fake_B = model_out(extract_feature_model, fake_B, False)
        vector_fake_A = model_out(extract_feature_model, fake_A, False)
        vector_real_B = model_out(extract_feature_model, B_batch_images, False)
        vector_real_A = model_out(extract_feature_model, A_batch_images, False)
        
        T = 4
        label_buff = tf.less(tf.abs(A_batch_labels - B_batch_labels), T)
        cast_label_buff = tf.cast(label_buff, tf.float32)  # loss를 좀더 추가해주어야 할거같음

        realA_fakeB_en = tf.reduce_sum(tf.abs(vector_real_A - vector_fake_B), 1)
        realA_fakeB_loss = (2/100) * (realA_fakeB_en*realA_fakeB_en) * cast_label_buff \
            + (1. - cast_label_buff) * 2*100*tf.exp(realA_fakeB_en*(-2.77/100))

        realB_fakeA_en = tf.reduce_sum(tf.abs(vector_real_B - vector_fake_A), 1)
        realB_fakeA_loss = (2/100) * (realB_fakeA_en*realB_fakeA_en) * cast_label_buff \
            + (1. - cast_label_buff) * 2*100*tf.exp(realB_fakeA_en*(-2.77/100))

        # A와 B 나이가 다르면 감소함수, 같으면 증가함수

        loss_buf = 0.
        loss_buf2 = 0.
        for j in range(FLAGS.batch_size):
            loss_buf += realA_fakeB_loss[j]
            loss_buf2 += realB_fakeA_loss[j]
        loss_buf /= FLAGS.batch_size
        loss_buf2 /= FLAGS.batch_size
        ################################################################################################

        Cycle_loss = (tf.reduce_mean(tf.abs(fake_A_ - A_batch_images)) \
            + tf.reduce_mean(tf.abs(fake_B_ - B_batch_images))) * 10.0
        G_gan_loss = tf.reduce_mean((DA_fake - tf.ones_like(DA_fake))**2) \
            + tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2)
        #print(Cycle_loss)
        Adver_loss = (tf.reduce_mean((DA_real - tf.ones_like(DA_real))**2) + tf.reduce_mean((DA_fake - tf.zeros_like(DA_fake))**2)) / 2. \
            + (tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2) + tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake))**2)) / 2.

        g_loss = Cycle_loss + G_gan_loss + loss_buf + loss_buf2
        d_loss = Adver_loss

    g_grads = g_tape.gradient(g_loss, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables)
    d_grads = g_tape.gradient(d_loss, A_discriminator.trainable_variables + B_discriminator.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, A_discriminator.trainable_variables + B_discriminator.trainable_variables))

    return g_loss, d_loss

def main():
    extract_feature_model = tf.keras.applications.VGG16(include_top=False,
                                                        input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    h = extract_feature_model.output
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(FLAGS.n_classes)(h)
    extract_feature_model = tf.keras.Model(inputs=extract_feature_model.input, outputs=h)
    extract_feature_model.summary()

    A2B_G_model = F2M_modelV8(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B2A_G_model = F2M_modelV8(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    A_discriminator = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B_discriminator = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    A2B_G_model.summary()
    A_discriminator.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                                   A_discriminator=A_discriminator, B_discriminator=B_discriminator,
                                   optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    if FLAGS.train:
        count = 0

        A_images = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_images = [FLAGS.A_img_path + data for data in A_images]
        A_labels = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        B_images = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_images = [FLAGS.B_img_path + data for data in B_images]
        B_labels = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        for epoch in range(FLAGS.epochs):
            # A 와 B의 비율을 매 epochs마다 랜덤으로 뿌려주면된다
            A_zip = list(zip(A_images, A_labels))
            shuffle(A_zip)
            B_zip = list(zip(B_images, B_labels))
            shuffle(B_zip)
            A_train_img = []
            A_train_lab = []
            B_train_img = []
            B_train_lab = []
            for i in range(FLAGS.age_range[0], FLAGS.age_range[1]):
                A_img_buf = []
                A_lab_buf = []
                B_img_buf = []
                B_lab_buf = []
                for j in range(len(A_images)):
                    if A_zip[j][1] == i:
                        A_img_buf.append(A_zip[j][0])
                        A_lab_buf.append(A_zip[j][1])

                for j in range(len(B_images)):
                    if B_zip[j][1] == i:
                        B_img_buf.append(B_zip[j][0])
                        B_lab_buf.append(B_zip[j][1])

                min_num = min(len(A_img_buf), len(B_img_buf))
                A_img_buf = A_img_buf[:min_num]
                B_img_buf = B_img_buf[:min_num]
                A_lab_buf = A_lab_buf[:min_num]
                B_lab_buf = B_lab_buf[:min_num]

                A_train_img.extend(A_img_buf)
                A_train_lab.extend(A_lab_buf)
                B_train_img.extend(B_img_buf)
                B_train_lab.extend(B_lab_buf)

            A_train_img, A_train_lab = np.array(A_train_img), np.array(A_train_lab)
            B_train_img, B_train_lab = np.array(B_train_img), np.array(B_train_lab)

            # 가까운 나이에 대해서 distance를 구하는 loss를 구성하면, 결국에는 해당이미지의 나이를 그대로 생성하는 효과?를 볼수있을것
            gener = tf.data.Dataset.from_tensor_slices(((A_train_img, A_train_lab), (B_train_img, B_train_lab)))
            gener = gener.shuffle(len(A_train_img))
            gener = gener.map(input_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_idx = len(A_train_img) // FLAGS.batch_size
            train_it = iter(gener)
            
            for step in range(train_idx):
                A_batch_images, A_batch_labels, B_batch_images, B_batch_labels = next(train_it)

                g_loss, d_loss = cal_loss(A2B_G_model, B2A_G_model, A_discriminator, B_discriminator,
                                          A_batch_images, A_batch_labels, B_batch_images, B_batch_labels,
                                          extract_feature_model)

                print(g_loss, d_loss)

                # 내일 그림으로 그려보자!

        a = 0


if __name__ == "__main__":
    main()
