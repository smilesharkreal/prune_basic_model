import numpy as np
image_size = 32
img_channels =3
from PIL import Image
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
    data_dir="../Cifar_10/cifar-10-batches-py"
    #data_dir = '../Datasets/Cifar_10/cifar-10-batches-py'
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')
    print(meta)
    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels,tr_l = load_data(train_files, data_dir, label_count)
    test_data, test_labels ,t_label= load_data(['test_batch'], data_dir, label_count)
    test_data,test_labels =order_test(test_data,t_label)

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

