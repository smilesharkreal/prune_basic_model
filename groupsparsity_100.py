# -*- coding:utf-8 -*-
import tensorflow as tf
import loadCifardata
import numpy as np
import time
import pickle
import os
import sys
import random
import matplotlib.pyplot as plt
import argparse
import h5py
from pandas.core.frame import DataFrame

class_num = 100
image_size = 32
img_channels = 3
iterations = 200
batch_size = 250
total_epoch = 164
weight_decay = 0.004
weight_decay1 = 0.003
weight_decay2 = 0.001
dropout_rate = 0.5
momentum_rate = 0.9

model_save_path = './vgg16_crossgroup_cifar100/model_ckpt'


def bias_variable(name,shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(name=name,initial_value=initial)
def threshold_variable(name,shape):
    initial = tf.ones(shape=shape)
    return tf.Variable(name=name,initial_value=initial,trainable=False)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)


def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)

def learning_rate_schedule(epoch_num):
    if epoch_num < 80:
        return 0.01
    elif epoch_num < 120:
        return 0.001
    else:
        return 0.0001


def run_testing(sess):
    acc = 0.0
    loss = 0.0
    lasso= 0.0
    pre_index = 0
    add = 1000
    for it in range(10):
        batch_x = test_x[pre_index:pre_index+add]
        batch_y = test_y[pre_index:pre_index+add]
        pre_index = pre_index + add
        loss_, acc_ ,lasoo= sess.run([cross_entropy, accuracy,l2],
                                feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})
        loss += loss_ / 10.0
        acc += acc_ / 10.0
        lasso += lasoo /10.0
    return acc, loss,lasso

def L2_peanlity(weight1):
    a = tf.multiply(weight1,weight1)
    b = tf.reduce_sum(a)
    b =tf.sqrt(b)
    return b

def L2_cross(weight1,weight2):
    a = tf.multiply(weight1,weight1)
    aa = tf.multiply(weight2,weight2)
    b = tf.reduce_sum(a)
    bb = tf.reduce_sum(aa)
    b =tf.sqrt(b+bb)
    return b
def grouplasso(weight1,weight_coff1,weight_coff2):
    split_conv1 = tf.split(weight1, num_or_size_splits=weight1.shape[3].value, axis=3)
    split_conv2 = tf.split(weight1, num_or_size_splits=weight1.shape[2].value, axis=2)
    glist = []
    clist = []
    for i in range(len(split_conv1)):
        glist.append(L2_peanlity(split_conv1[i]))
    for i in range(len(split_conv2)):
        clist.append(L2_peanlity(split_conv2[i]))
    t1 = glist[0]
    t2 = clist[0]
    for i in range(len(clist)-1):
        t2 += clist[i+1]
    for i in range(len(glist)-1):
        t1 += glist[i+1]
    s = t1* weight_coff1
    s += t2*weight_coff2
    return s
def groupcrosslasso(weight1,weight2,weight_coff):
    split_conv1 = tf.split(weight1, num_or_size_splits=weight1.shape[3].value, axis=3)
    split_conv2 = tf.split(weight2, num_or_size_splits=weight2.shape[2].value, axis=2)
    list = []
    for i in range(len(split_conv1)):
        list.append(L2_cross(split_conv1[i],split_conv2[i]))
    s = list[0]
    for i in range(len(list) - 1):
        s = s + list[i + 1]
    s = s * weight_coff
    return s

def vgg16(x, keep_prob):
    tensors = []
    # build_network
    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable("bias1_1",[64])
    t_conv1_1 = threshold_variable("tconv1_1",shape=[3, 3, 3, 64])
    t_bias1_1 = threshold_variable("tbias1_1",shape=[64])
    tensors += [W_conv1_1, b_conv1_1,t_conv1_1,t_bias1_1]
    output = tf.nn.relu(batch_norm(conv2d(x, W_conv1_1*t_conv1_1)+ b_conv1_1*t_bias1_1))

    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_2 = bias_variable("bias1_2",[64])
    t_conv1_2 = threshold_variable("tconv1_2", shape=[3, 3, 64, 64])
    t_bias1_2 = threshold_variable("tbias1_2", shape=[64])
    tensors += [W_conv1_2, b_conv1_2,t_conv1_2,t_bias1_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2*t_conv1_2) + b_conv1_2*t_bias1_2))
    output = max_pool(output, 2, 2, "pool1")
    # out :16

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable("bias2_1",[128])
    t_conv2_1 = threshold_variable("tconv2_1", shape=[3, 3, 64, 128])
    t_bias2_1 = threshold_variable("tbias2_1", shape=[128])
    tensors += [W_conv2_1, b_conv2_1,t_conv2_1,t_bias2_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1*t_conv2_1) + b_conv2_1*t_bias2_1))


    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_2 = bias_variable("bias2_2",[128])
    t_conv2_2 = threshold_variable("tconv2_2", shape=[3, 3, 128, 128])
    t_bias2_2 = threshold_variable("tbias2_2", shape=[128])
    tensors += [W_conv2_2, b_conv2_2,t_conv2_2,t_bias2_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2*t_conv2_2) + b_conv2_2*t_bias2_2))
    output = max_pool(output, 2, 2, "pool2")
    # out :8

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable("bias3_1",[256])
    t_conv3_1 = threshold_variable("tconv3_1", shape=[3, 3, 128, 256])
    t_bias3_1 = threshold_variable("tbias3_1", shape=[256])
    tensors += [W_conv3_1, b_conv3_1,t_conv3_1,t_bias3_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_1*t_conv3_1) + b_conv3_1*t_bias3_1))
    # output = tf.nn.relu(conv2d(output, W_conv3_1) + b_conv3_1)

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = bias_variable("bias3_2",[256])
    t_conv3_2 = threshold_variable("tconv3_2", shape=[3, 3, 256, 256])
    t_bias3_2 = threshold_variable("tbias3_2", shape=[256])
    tensors += [W_conv3_2, b_conv3_2,t_conv3_2,t_bias3_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2*t_conv3_2) + b_conv3_2*t_bias3_2))
    # output = tf.nn.relu(conv2d(output, W_conv3_2) + b_conv3_2)

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = bias_variable("bias3_3",[256])
    t_conv3_3 = threshold_variable("tconv3_3", shape=[3, 3, 256, 256])
    t_bias3_3 = threshold_variable("tbias3_3", shape=[256])
    tensors += [W_conv3_3, b_conv3_3,t_conv3_3,t_bias3_3]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_3*t_conv3_3) + b_conv3_3*t_bias3_3))
    # output = tf.nn.relu(conv2d(output, W_conv3_3) + b_conv3_3)
    output = max_pool(output, 2, 2, "pool3")
    # out :4

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable("bias4_1",[512])
    t_conv4_1 = threshold_variable("tconv4_1", shape=[3, 3, 256, 512])
    t_bias4_1 = threshold_variable("tbias4_1", shape=[512])
    tensors += [W_conv4_1, b_conv4_1,t_conv4_1,t_bias4_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_1*t_conv4_1) + b_conv4_1*t_bias4_1))
    # output = tf.nn.relu(conv2d(output, W_conv4_1) + b_conv4_1)

    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = bias_variable("bias4_2",[512])
    t_conv4_2 = threshold_variable("tconv4_2", shape=[3, 3, 512, 512])
    t_bias4_2 = threshold_variable("tbias4_2", shape=[512])
    tensors += [W_conv4_2, b_conv4_2,t_conv4_2,t_bias4_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2*t_conv4_2) + b_conv4_2*t_bias4_2))
    # output = tf.nn.relu(conv2d(output, W_conv4_2) + b_conv4_2)

    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = bias_variable("bias4_3",[512])
    t_conv4_3 = threshold_variable("tconv4_3", shape=[3, 3, 512, 512])
    t_bias4_3 = threshold_variable("tbias4_3", shape=[512])
    tensors += [W_conv4_3, b_conv4_3,t_conv4_3,t_bias4_3]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_3*t_conv4_3) + b_conv4_3*t_bias4_3))
    # output = tf.nn.relu(conv2d(output, W_conv4_3) + b_conv4_3)
    output = max_pool(output, 2, 2)
    # out :2

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable("bias5_1",[512])
    t_conv5_1 = threshold_variable("tconv5_1", shape=[3, 3, 512, 512])
    t_bias5_1 = threshold_variable("tbias5_1", shape=[512])
    tensors += [W_conv5_1, b_conv5_1,t_conv5_1,t_bias5_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_1*t_conv5_1) + b_conv5_1*t_bias5_1))
    # output = tf.nn.relu(conv2d(output, W_conv5_1) + b_conv5_1)

    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = bias_variable("bias5_2",[512])
    t_conv5_2 = threshold_variable("tconv5_2", shape=[3, 3, 512, 512])
    t_bias5_2 = threshold_variable("tbias5_2", shape=[512])
    tensors += [W_conv5_2, b_conv5_2,t_conv5_2,t_bias5_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2*t_conv5_2) + b_conv5_2*t_bias5_2))
    # output = tf.nn.relu(conv2d(output, W_conv5_2) + b_conv5_2)

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_3 = bias_variable("bias5_3",[512])
    t_conv5_3 = threshold_variable("tconv5_3", shape=[3, 3, 512, 512])
    t_bias5_3 = threshold_variable("tbias5_3", shape=[512])
    tensors += [W_conv5_3, b_conv5_3,t_conv5_3,t_bias5_3]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_3*t_conv5_3) + b_conv5_3*t_bias5_3))
    # output = tf.nn.relu(conv2d(output, W_conv5_3) + b_conv5_3)
    # output = max_pool(output, 2, 2)

    # output = tf.contrib.layers.flatten(output)
    flatten_output = tf.reshape(output, [-1, 2 * 2 * 512])

    W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable("fc1_b",[4096])
    output = tf.nn.relu(batch_norm(tf.matmul(flatten_output, W_fc1) + b_fc1))
    # output = tf.nn.relu(tf.matmul(flatten_output, W_fc1) + b_fc1)
    output = tf.nn.dropout(output, keep_prob)

    W_fc2 = tf.get_variable('fc2', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable('fc2_b',[4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2))
    # output = tf.nn.relu(tf.matmul(output, W_fc2) + b_fc2)
    output = tf.nn.dropout(output, keep_prob)

    W_fc3 = tf.get_variable('fc3', shape=[4096, 100], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc3 = bias_variable('fc3_b',[100])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3))
    for i in range(12):
        #tf.add_to_collection("losses",tf.nn.l2_loss(tensors[4 * i]*tensors[4 * i+2])*weight_decay)
        #tf.add_to_collection('losses', groupcrosslasso(tensors[4 * i]*tensors[4 * i+2], tensors[4*(i+1)]*tensors[4*(i+1)+2],weight_decay))
        tf.add_to_collection('losses', grouplasso(tensors[4 * i] * tensors[4 * i + 2], weight_decay1, weight_decay2))
    return output, tensors

def sort_index(weight):
    weight = np.abs(weight)
    weight = np.sum(weight,axis=(0,1,2))
    index = np.argsort(weight)
    return index


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    c100 = loadCifardata.Cifar("../Datasets/Cifar-10/cifar-100-python")
    train_x, train_y, test_x, test_y = c100.prepare_data()
    train_x, test_x = loadCifardata.data_preprocessing(train_x, test_x)
    # 返回vgg16 输出，tensors是保存的tensors常量 train_x, train_y, test_x, test_y 数据预处理过的
    output, tensors= vgg16(x, keep_prob)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 )
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:

        modelvalid = False
        if modelvalid == True:
            ckpt = tf.train.get_checkpoint_state('./vgg16_crossgroup_cifar100/')
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            # ckpt = tf.train.latest_checkpoint("./vgg16_crossgroup/")
            # saver.restore(sess,ckpt)
            acc, loss, lasso1 = run_testing(sess)
            print("acc is ", acc)
            print("loss %.4f, cglasso is %.4f" % (loss, lasso1))
            tensors_ = sess.run(tensors)
            zore = 0
            flag = 0
            for i in range(13):
                i0 = 0
                i1 = 0
                t = np.abs(tensors_[4 * i])
                t = np.sum(t, axis=(0, 1, 2))
                for j in t:
                    if j <= 0.1:
                        i1 += 1
                        if j == 0:
                            i0 += 1
                print("the %d layer the number of 0 is %d, the number of 0.1 is %d" % (i + 1, i0, i1))
                zore += i0
                flag += i1
            print("total the number of 0 is %d, the number of 0.1 is %d" % (zore, flag))
            f =h5py.File('./crossgroup_cifar100.h5','w')
            for i in range(13):
                str_label = "weight"+str(i)
                str_bias = "bias"+str(i)
                f[str_label] =tensors_[4*i]
                f[str_bias] = tensors_[4*i +1]
            f.close()
            print("done")

        modeltrain = False
        if modeltrain == True:
            test_acc = []
            cross = []
            change = []
            train_acc_ = []
            train_loss_ = []
            train_l2 = []
            lasso1 = []
            lasso2 = []
            lr_ = []
            sess.run(tf.global_variables_initializer())
            # ckpt = tf.train.latest_checkpoint("./vgg16_crossgroup_0004_160/")
            # saver.restore(sess, ckpt)
            # initialize_uninitialized(sess)
            # this is train
            # tensors_ = sess.run(tensors)
            # f = h5py.File('./crossgroup_0005.h5', 'w')
            # for i in range(13):
            #     str_label = "weight" + str(i)
            #     str_bias = "bias" + str(i)
            #     f[str_label] = tensors_[4 * i]
            #     f[str_bias] = tensors_[4 * i + 1]
            # f.close()
            # print("done")
            max_acc = 0
            for ep in range(1, total_epoch + 1):
                lr = learning_rate_schedule(ep)
                lr_.append(lr)
                pre_index = 0
                train_acc = 0.0
                train_loss = 0.0
                l2_loss = 0.0
                start_time = time.time()

                # f.write("epoch %d/%d:\n" % (ep, total_epoch))
                for it in range(1, iterations + 1):
                    if pre_index + batch_size >50000:
                        next_index = 50000 -(pre_index + batch_size)
                        batch_x = train_x[next_index:]
                        batch_y = train_y[next_index:]
                        batch_x_ = train_x[:batch_size+next_index]
                        batch_y_ = train_y[:batch_size+next_index]
                        batch_x = np.append(batch_x,batch_x_)
                        batch_x = batch_x.reshape([batch_size,image_size,image_size,img_channels])
                        batch_y = np.append(batch_y,batch_y_)
                        batch_y = batch_y.reshape([batch_size,class_num])
                    else:
                        batch_x = train_x[pre_index:pre_index + batch_size]
                        batch_y = train_y[pre_index:pre_index + batch_size]
                    batch_x = loadCifardata.data_augmentation(batch_x)
                    _, batch_loss, _l2= sess.run([train_step, cross_entropy,l2],
                                             feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_rate,
                                                        learning_rate: lr, train_flag: True})
                    batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})
                    train_loss += batch_loss
                    train_acc += batch_acc
                    l2_loss += _l2
                    pre_index += batch_size
                    # 单次训练结束
                    if it == iterations:
                        train_loss /= iterations
                        train_acc /= iterations
                        l2_loss /= iterations
                        val_acc, val_loss, l2_ = run_testing(sess)
                        # print(ep, "the acc is", val_acc)
                        train_acc_.append(train_acc)
                        train_loss_.append(train_loss)
                        train_l2.append(l2_loss)
                        test_acc.append(val_acc)
                        lasso1.append(l2_)
                        cross.append(val_loss)
                        if max_acc < val_acc and val_acc >= 0.60:
                            saver.save(sess, './vgg16_groupsparsity/model_ckpt', ep)
                            max_acc = val_acc
                        print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, groupcross_lasso: %.4f"
                              "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f\n"
                              % (
                                  ep, total_epoch, int(time.time() - start_time), train_loss, l2_, train_acc, val_loss,
                                  val_acc))
                    # if ep == 120:
                    #     saver.save(sess, './vgg16_crossgroup_0004_120/model_ckpt', ep)
                    # if ep == 160:
                    #     saver.save(sess, './vgg16_crossgroup_0004_160/model_ckpt', ep)
            np.save("./cg/sparsity_test_l2loss", lasso1)
            np.save("./cg/sparsity_test_acc", test_acc)
            np.save("./cg/sparsity_test_entroyloss", cross)
            np.save("./cg/sparsity_train_acc", train_acc_)
            np.save("./cg/sparsity_train_entriyloss", train_loss_)
            np.save("./cg/sparsity_train_l2loss", train_l2)
            np.save("./cg/sparsity_train_lr", lr_)
            print("the train process is done!,and the max acc is ",max_acc)
            ckpt = tf.train.latest_checkpoint("./vgg16_groupsparsity/")
            saver.restore(sess, ckpt)
            tensors_ = sess.run(tensors)
            f =h5py.File('./groupsparsity_cifar100.h5','w')
            for i in range(13):
                str_label = "weight"+str(i)
                str_bias = "bias"+str(i)
                f[str_label] =tensors_[4*i]
                f[str_bias] = tensors_[4*i +1]
            f.close()
            print("done")
        modelcut = True
        if modelcut == True:
            step_acc = []
            with tf.Session() as sess:
                ckpt = tf.train.latest_checkpoint("./vgg16_groupsparsity/")
                saver.restore(sess, ckpt)
                acc, loss, pel = run_testing(sess)
                print("the restore acc is ", acc)
                tensors_ = sess.run(tensors)
                # f =h5py.File('./crossgroup_0004.h5','w')
                # for i in range(13):
                #     str_label = "weight"+str(i)
                #     str_bias = "bias"+str(i)
                #     f[str_label] =tensors_[4*i]
                #     f[str_bias] = tensors_[4*i +1]
                # f.close()
                # print("done")
                p = 0.1
                zore = 0
                oneflag = 0
                for i in range(13):
                    s_time = time.time()
                    weight_ = tensors_[4 * i]
                    th_ = tensors_[4 * i + 2].copy()
                    b_ = tensors_[4 * i + 3].copy()
                    if i <= 11:
                        th2_ = tensors_[4 * (i + 1) + 2].copy()
                    weight_ = np.reshape(weight_, [-1, weight_.shape[-1]])
                    weight_ = np.abs(weight_)
                    weight_ = np.sum(weight_, axis=0)
                    i0 = 0
                    i1 = 0
                    # if i == 0:
                    #     p = 0
                    # else:
                    #     p = 0.1
                    for j in range(len(weight_)):
                        if weight_[j] <= p:
                            th_[:, :, :, j] = 0
                            if i <= 11:
                                th2_[:, :, j, :] = 0
                            b_[j] = 0
                            i1 += 1
                            if weight_[j] == 0:
                                i0 += 1
                    sess.run(tf.assign(tensors[4 * i + 2], th_))
                    sess.run(tf.assign(tensors[4 * i + 3], b_))
                    if i <= 11:
                        sess.run(tf.assign(tensors[4 * (i + 1) + 2], th2_))
                    zore += i0
                    oneflag += i1
                    acc, loss, pel = run_testing(sess)
                    print("the %dth layer has %d filter is zero,"
                          "and the number of %d less %f will be prune,and after it's acc %.4f,"
                          "the process consume time is %.4f" % (
                              i + 1, i0, i1, p, acc, time.time() - s_time))
                print(
                    "after the 13 layer prune ,the convolution has %d filter is zore ,"
                    "and the number of %d less %f will be prune,and after it's acc %.4f" % (
                        zore, oneflag, p, acc))



