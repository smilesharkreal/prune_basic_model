import tensorflow as tf
import numpy as np
import random
import time
image_size =32
img_channels =3
class_num = 10
momentum_rate =0.9
weight_decay = 0.0001
total_epoch = 164
batch_size =250
iterations = 200
model_save_path = './dynamic_resNet20_model/modl_ckpt'
# 变量的定义
# 读取序列化数据 返回类型字典类型数据
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# 对 label进行处理
def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    t_label = labels
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])  # 将图片转置 跟卷积层一致
    return data, labels,t_label
# 将数据导入
def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels
# 测试集排序
def order_test(test_data,labels):
    a = []
    b = []
    for i in range(10):
        a.append([])
    for i in range(10000):
        a[int(labels[i])].append(test_data[i, :, :, :])
        c = i // 1000
        b.append(c)
    b = np.array([[float(i == label) for i in range(10)] for label in b])
    a = np.array(a)
    a = a.reshape([-1, a.shape[-3], a.shape[-2], a.shape[-1]])
    return a,b
def prepare_data():
    print("======Loading data======")
    #download_data()
    data_dir="../Datasets/Cifar_10/cifar-10-batches-py"
    #data_dir = '../Datasets/Cifar_10/cifar-10-batches-py'
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')
    print(meta)
    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels,tr_l = load_data(train_files, data_dir, label_count)
    test_data, test_labels ,t_label= load_data(['test_batch'], data_dir, label_count)
    # test_data,test_labels =order_test(test_data,t_label)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")
    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))  # 打乱数组
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")
    return train_data, train_labels, test_data, test_labels
## Z-score 标准化
def data_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# flag_like
def weight_flag(conv):
    return tf.Variable(tf.ones_like(conv),trainable=False)
# 卷积定义
def conv2d(input, filter, strides, padding="SAME", name=None):
    return tf.nn.conv2d(input, filter, strides, padding="SAME", name=name)  # padding="SAME"用零填充边界
def residual_block(x, output_channel,k, is_training,para,para_flag):
    '''
    定义残差块儿
    :param x: 输入tensor
    :param output_channel: 输出的通道数
    :return: tensor
    需要注意的是:每经过一个stage,通道数就要 * 2
    在同一个stage中,通道数是没有变化的
    '''
    input_channel = x.get_shape().as_list()[-1] # 拿出 输入 tensor 的 最后一维:也就是通道数
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = [1,2,2,1] #
    elif input_channel == output_channel:
        increase_dim = False
        strides = [1,1,1,1]
    else:
        raise Exception("input channel can't match output channel")

    kernel_1 = weight_variable(shape=[3,3,input_channel,output_channel])
    kernel_1_flag = weight_flag(kernel_1)
    para_flag.append(kernel_1_flag)
    bias_1 = weight_variable(shape=[output_channel])
    para.append(kernel_1)
    para.append(bias_1)
    conv_1 = conv2d(x, tf.multiply(kernel_1,kernel_1_flag), strides=strides) + bias_1
    bn_1 = tf.layers.batch_normalization(conv_1, training=is_training)
    conv_1 =tf.nn.relu(bn_1)

    kernel_2 = weight_variable(shape=[3, 3, conv_1.get_shape().as_list()[-1], output_channel])
    kernel_2_flag = weight_flag(kernel_2)
    para_flag.append(kernel_2_flag)
    bias_2 = weight_variable(shape=[output_channel])
    para.append(kernel_2)
    para.append(bias_2)
    conv_2 = conv2d(conv_1, tf.multiply(kernel_2,kernel_2_flag), strides=[1,1,1,1]) + bias_2
    bn_2 = tf.layers.batch_normalization(conv_2, training=is_training)
    # conv_2 = tf.nn.relu(bn_2)
    if increase_dim: # 需要使用降采样
        # pooled_x 数据格式 [ None, image_width, image_height, channel ]
        # 要求格式 [ None, image_width, image_height, channel * 2 ]
        pooled_x = tf.layers.average_pooling2d(x,(2, 2), (2, 2),padding = 'valid')
        '''
        如果输出通道数是输入的两倍的话,需要增加通道数量.
        maxpooling 只能降采样,而不能增加通道数,
        所以需要单独增加通道数
        '''
        padded_x = tf.pad(pooled_x, # 参数 2 ,在每一个通道上 加 pad
                           [
                               [ 0, 0 ],
                               [ 0, 0 ],
                               [ 0, 0 ],
                               [input_channel // 2, input_channel // 2] # 实际上就是 2倍input_channel,需要均分开
                            ]
                          )
    else:
        padded_x = x
    output_x = bn_2 + padded_x   # 就是 公式: H(x) = F(x) + x
    output_x = tf.nn.relu(output_x)
    return  output_x
def res_net(x, num_residual_blocks, num_filter_base, class_num, is_training):
    para = []
    para_flag = []
    layers = []  # 保存每一个残差块的输出
    with tf.name_scope("conv1"):
        kernel_1=weight_variable([3,3,3,16])
        kernel_1_flag = weight_flag(kernel_1)
        para_flag.append(kernel_1_flag)
        bias_1 = weight_variable([16])
        para.append(kernel_1)
        para.append(bias_1)
        conv_1 = conv2d(x,tf.multiply(kernel_1,kernel_1_flag),strides=[1,1,1,1])+bias_1
        bn = tf.layers.batch_normalization(conv_1, training=is_training)
        layer_1 =tf.nn.relu(bn)
        #layer_1 = tf.nn.relu(conv2d(x,kernel_1,strides=[1,1,1,1])+bias_1,name='layer_1')
        layers.append(layer_1)
    k =0
    for i in range(2,5):
        for j in range(1,num_residual_blocks+1):
            k += 2
            with tf.name_scope('conv%d_%d'%(i,j)):
                conv = residual_block(layers[-1],num_filter_base * (2 ** (i-2)),k,is_training,para,para_flag)
                layers.append(conv)
    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])  # 求平均值函数,参数二 指定 axis
        logits = tf.layers.dense(global_pool, class_num)
        layers.append(logits)
    return layers[-1],para,para_flag
# 随机裁剪
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


# 对数据随机左右翻转
def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch
def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch
def learning_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001
def run_testing(sess):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    add = 1000
    for it in range(10):
        batch_x = test_x[pre_index:pre_index+add]
        batch_y = test_y[pre_index:pre_index+add]
        pre_index = pre_index + add
        loss_, acc_  = sess.run([cross_entropy, accuracy],
                                feed_dict={x: batch_x, y_: batch_y, is_training: False})
        loss += loss_ / 10.0
        acc += acc_ / 10.0
    return acc, loss

def filter_shuffle(conv,g):
    n, h, w, c = conv.shape.as_list()
    x_reshaped = tf.reshape(conv, [-1, h, w, g, c // g])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])
    return output
# 对filter 进行随机分配
def filter_pool(conv,g,sess):
    conv = filter_shuffle(conv,g)
    # c_conv = tf.reshape(conv,[-1,conv.shape.as_list()[-1]])
    # mean,var = tf.nn.moments(c_conv,axes=0)
    # filter_var_mean = tf.reduce_mean(var)
    list_gconv = tf.split(conv,g,axis=-1)
    n, h, w, c = list_gconv[0].shape.as_list()
    flag_list = []
    for i in range(g):
        split_c_conv = tf.reshape(list_gconv[i],[-1,list_gconv[i].shape.as_list()[-1]])
        # 求得每一个group中的方差
        split_m,split_v = tf.nn.moments(split_c_conv,axes=0)
        # 求得一个group中方差的平均值
        # split_v_mean = tf.reduce_mean(split_v)
        t = tf.argmax(split_v)
        t = sess.run(t)
        for j in range(c):
            if j == t:
                flag_list.append(tf.ones([n,h,w,1]))
            else:
                flag_list.append(tf.zeros([n,h,w,1]))
    flag_conv = tf.concat(flag_list,axis=-1)
    return flag_conv
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = data_preprocessing(train_x, test_x)

    output,para,para_flag = res_net(x,3,16,10,is_training)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True). \
            minimize(cross_entropy+l2*weight_decay)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(max_to_keep=2)
    with tf.Session() as sess:
        #this is train
        flag =0
        if flag == 1:
            sess.run(tf.global_variables_initializer())
            max_acc = 0
            f = open("./pool_resnet20_log.txt", "a")
            # f =open("./log_resnet32_acc.txt","a")
            # for ep in range(1, total_epoch + 1):
            #     lr = learning_rate_schedule(ep)
            #     pre_index = 0
            #     train_acc = 0.0
            #     train_loss = 0.0
            #     start_time = time.time()
            #     f.write("epoch %d/%d:\n" % (ep, total_epoch))
            #     for it in range(1, iterations + 1):
            #         batch_x = train_x[pre_index:pre_index + batch_size]
            #         batch_y = train_y[pre_index:pre_index + batch_size]
            #         batch_x = data_augmentation(batch_x)
            #         _, batch_loss,batch_acc = sess.run([train_step, cross_entropy,accuracy],
            #                                  feed_dict={x: batch_x, y_: batch_y, learning_rate: lr, is_training: True})
            #         # batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y,  is_training: True})
            #         train_loss += batch_loss
            #         train_acc += batch_acc
            #         pre_index += batch_size
            #         # 单次训练结束
            #         if it == iterations:
            #             train_loss /= iterations
            #             train_acc /= iterations
            #             val_acc, val_loss= run_testing(sess)
            #             f.write("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
            #                   "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f\n"
            #                   % (
            #                       it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
            #             print("the epoch is %d ,and the test accuracy is %f"%(ep,val_acc))
            #             if max_acc<val_acc:
            #                 saver.save(sess, model_save_path, ep)
            #                 max_acc =val_acc
            #         else:
            #             f.write("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f\n"
            #                   % (it, iterations, train_loss / it, train_acc / it))
            #     if max_acc>0.95:
            #         f.write("the train is done!")
            #         break

            ## restore
            t = 0
            for i in range(total_epoch + 1):
                T = time.time()
                for j in range(19):
                    g = (para[j * 2].shape.as_list()[-1]) // 2
                    conv_flag = filter_pool(para[j * 2], g, sess)
                    #para_flag[j] = conv_flag
                print("filter time is %d", int(time.time() - T))
                lr = learning_rate_schedule(i)
                pre_index = 0
                train_acc = 0.0
                train_loss = 0.0

                f.write("epoch %d/%d:\n" % (i, total_epoch))
                for it in range(1, iterations + 1):
                    start_time = time.time()
                    batch_x = train_x[pre_index:pre_index + batch_size]
                    batch_y = train_y[pre_index:pre_index + batch_size]
                    batch_x = data_augmentation(batch_x)
                    _, batch_loss, batch_acc = sess.run([train_step, cross_entropy, accuracy],
                                                        feed_dict={x: batch_x, y_: batch_y, learning_rate: lr,
                                                                   is_training: True})
                    train_loss += batch_loss
                    train_acc += batch_acc
                    pre_index += batch_size
                    # 单次训练结束
                    if it == iterations:
                        train_loss /= iterations
                        train_acc /= iterations
                        val_acc, val_loss = run_testing(sess)
                        f.write("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                                "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f\n"
                                % (
                                    it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss,
                                    val_acc))
                        print("the epoch is %d ,and the test accuracy is %f" % (i, val_acc))
                        if max_acc < val_acc:
                            saver.save(sess, model_save_path, i)
                            max_acc = val_acc
                    else:
                        f.write("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f\n"
                                % (it, iterations, train_loss / it, train_acc / it))
                if max_acc > 0.90:
                    f.write("the train is done!")
                    f.close()
                    break
        else:
            #  restore this model
            ckpt = tf.train.get_checkpoint_state('./dynamic_resNet20_model')
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            global_step = ckpt.all_model_checkpoint_paths[-1].split('/')[-1].split('-')[-1]
            val_acc, val_loss = run_testing(sess)
            print("the model's accuracy is %4f" % (val_acc))
            start_time = time.time()
            para = sess.run(para)
            print("sess.run time is ", float(time.time() - start_time))
            np.savez("./para",para[0],para[2],para[4],para[6],para[8],para[10],para[12],para[14],para[16],para[18],
                      para[20],para[22],para[24],para[26],para[28],para[30],para[32],para[34],para[36])
            print("save time is ",float(time.time()-start_time))