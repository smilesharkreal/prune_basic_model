import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class Lenet_Model(object):
    def __init__(self, x, keep_prob, is_training, config_params):
        # para = []
        shape = []
        self.config = config_params
        model_channel = config_params["Lenet_channel"]

        # 卷积定义
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def bias_variable(name, shape):
            initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
            return tf.Variable(name=name, initial_value=initial)

        def max_pool(input, k_size=1, stride=1, name=None):
            return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                                  padding='SAME', name=name)
        # 32*32
        with tf.name_scope("conv1"):
            W_conv1_1 = tf.get_variable('conv1', shape=[3, 3, 3, model_channel[0]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv1_1 = bias_variable("bias1_1", [model_channel[0]])
            tf.add_to_collection("weight", W_conv1_1)
            tf.add_to_collection("bias", b_conv1_1)
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(x, W_conv1_1) + b_conv1_1, training=is_training))
            shape.append(output.get_shape().as_list())
            shape.append(W_conv1_1.get_shape().as_list())
        # 16*16
        with tf.name_scope("conv2"):
            W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, model_channel[0], model_channel[1]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv1_2 = bias_variable("bias1_2", [model_channel[1]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(conv2d(output, W_conv1_2) + b_conv1_2, training=is_training))
            tf.add_to_collection("weight", W_conv1_2)
            tf.add_to_collection("bias", b_conv1_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv1_2.get_shape().as_list())
            output = max_pool(output, 2, 2, "pool1")
            # out :16
        # 8*8
        with tf.name_scope("conv3"):
            W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, model_channel[1], model_channel[2]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv2_1 = bias_variable("bias2_1", [model_channel[2]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(conv2d(output, W_conv2_1) + b_conv2_1, training=is_training))
            tf.add_to_collection("weight", W_conv2_1)
            tf.add_to_collection("bias", b_conv2_1)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv2_1.get_shape().as_list())
        # 4*4
        with tf.name_scope("conv4"):
            W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, model_channel[2], model_channel[3]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv2_2 = bias_variable("bias2_2", [model_channel[3]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(conv2d(output, W_conv2_2) + b_conv2_2, training=is_training))
            tf.add_to_collection("weight", W_conv2_2)
            tf.add_to_collection("bias", b_conv2_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv2_2.get_shape().as_list())
            output = max_pool(output, 2, 2, "pool2")
        #     # out :8
        # with tf.name_scope("conv5"):
        #     W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, model_channel[3], model_channel[4]],
        #                                 initializer=tf.keras.initializers.he_normal())
        #     b_conv3_1 = bias_variable("bias3_1", [model_channel[4]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv3_1) + b_conv3_1, training=is_training))
        #     tf.add_to_collection("weight", W_conv3_1)
        #     tf.add_to_collection("bias", b_conv3_1)
        #     shape.append(output.get_shape().as_list())
        #     shape.append(W_conv3_1.get_shape().as_list())
        #
        # with tf.name_scope("conv6"):
        #     W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, model_channel[4], model_channel[5]],
        #                                 initializer=tf.keras.initializers.he_normal())
        #     b_conv3_2 = bias_variable("bias3_2", [model_channel[5]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv3_2) + b_conv3_2, training=is_training))
        #     tf.add_to_collection("weight", W_conv3_2)
        #     tf.add_to_collection("bias", b_conv3_2)
        #     shape.append(output.get_shape().as_list())
        #     shape.append(W_conv3_2.get_shape().as_list())
        # with tf.name_scope("conv7"):
        #     W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3,model_channel[5], model_channel[6]],
        #                                 initializer=tf.keras.initializers.he_normal())
        #     b_conv3_3 = bias_variable("bias3_3", [model_channel[6]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv3_3) + b_conv3_3, training=is_training))
        #     tf.add_to_collection("weight", W_conv3_3)
        #     tf.add_to_collection("bias", b_conv3_3)
        #     shape.append(output.get_shape().as_list())
        #     shape.append(W_conv3_3.get_shape().as_list())
        #     output = max_pool(output, 2, 2, "pool3")
        #     # out :4
        # with tf.name_scope("conv8"):
        #     W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3,model_channel[6], model_channel[7]],
        #                                 initializer=tf.keras.initializers.he_normal())
        #     b_conv4_1 = bias_variable("bias4_1", [model_channel[7]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv4_1) + b_conv4_1, training=is_training))
        #     tf.add_to_collection("weight", W_conv4_1)
        #     tf.add_to_collection("bias", b_conv4_1)
        #     shape.append(output.get_shape().as_list())
        #     shape.append(W_conv4_1.get_shape().as_list())
        # with tf.name_scope("conv9"):
        #     W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3,model_channel[7],model_channel[8]],
        #                                 initializer=tf.keras.initializers.he_normal())
        #     b_conv4_2 = bias_variable("bias4_2", [model_channel[8]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv4_2) + b_conv4_2, training=is_training))
        #     tf.add_to_collection("weight", W_conv4_2)
        #     tf.add_to_collection("bias", b_conv4_2)
        #     shape.append(output.get_shape().as_list())
        #     shape.append(W_conv4_2.get_shape().as_list())
        # with tf.name_scope("conv10"):
        #     W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, model_channel[8], model_channel[9]],
        #                                 initializer=tf.keras.initializers.he_normal())
        #     b_conv4_3 = bias_variable("bias4_3", [model_channel[9]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv4_3) + b_conv4_3, training=is_training))
        #     tf.add_to_collection("weight", W_conv4_3)
        #     tf.add_to_collection("bias", b_conv4_3)
        #     shape.append(output.get_shape().as_list())
        #     shape.append(W_conv4_3.get_shape().as_list())
        #     output = max_pool(output, 2, 2)
        #     # out :2
        # with tf.name_scope("conv11"):
        #     W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, model_channel[9], model_channel[10]],
        #                                 initializer=tf.keras.initializers.he_normal())
        #     b_conv5_1 = bias_variable("bias5_1", [model_channel[10]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv5_1) + b_conv5_1, training=is_training))
        #     tf.add_to_collection("weight", W_conv5_1)
        #     tf.add_to_collection("bias", b_conv5_1)
        #     shape.append(output.get_shape().as_list())
        #     shape.append(W_conv5_1.get_shape().as_list())
        # with tf.name_scope("conv12"):
        #     W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, model_channel[10], model_channel[11]],
        #                                 initializer=tf.keras.initializers.he_normal())
        #     b_conv5_2 = bias_variable("bias5_2", [model_channel[11]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv5_2) + b_conv5_2, training=is_training))
        #     tf.add_to_collection("weight", W_conv5_2)
        #     tf.add_to_collection("bias", b_conv5_2)
        #     shape.append(output.get_shape().as_list())
        #     shape.append(W_conv5_2.get_shape().as_list())
        # 2*2
        with tf.name_scope("conv5"):
            W_conv5_3 = tf.get_variable('conv5', shape=[3, 3, model_channel[3], model_channel[4]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv5_3 = bias_variable("bias5", [model_channel[12]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv5_3) + b_conv5_3, training=is_training))
            tf.add_to_collection("weight", W_conv5_3)
            tf.add_to_collection("bias", b_conv5_3)
            # 池化
            output = max_pool(output, 2, 2)

            shape.append(output.get_shape().as_list())
            shape.append(W_conv5_3.get_shape().as_list())

            # output = tf.contrib.layers.flatten(output)
            flatten_output = tf.reshape(output, [-1, 1 * 1 * model_channel[4]])
        with tf.name_scope("fc1"):
            W_fc1 = tf.get_variable('fc1', shape=[model_channel[4], model_channel[5]], initializer=tf.keras.initializers.he_normal())
            b_fc1 = bias_variable("fc1_b", [model_channel[5]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(flatten_output, W_fc1) + b_fc1, training=is_training))

            output = tf.nn.dropout(output, keep_prob)
        # with tf.name_scope("fc2"):
        #     W_fc2 = tf.get_variable('fc2', shape=[model_channel[13], model_channel[14]], initializer=tf.keras.initializers.he_normal())
        #     b_fc2 = bias_variable('fc2_b', [model_channel[14]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(output, W_fc2) + b_fc2, training=is_training))
        #     output = tf.nn.dropout(output, keep_prob)
        with tf.name_scope("fc2"):
            W_fc3 = tf.get_variable('fc2', shape=[model_channel[5], self.config["class_num"]],
                                    initializer=tf.keras.initializers.he_normal())
            b_fc3 = bias_variable('fc2_b', [self.config["class_num"]])
            output = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(output, W_fc3) + b_fc3, training=is_training))
            self._out = output
            self._shape = shape

    @property
    def output(self):
        return self._out

    @property
    def output_shape(self):
        return self._shape