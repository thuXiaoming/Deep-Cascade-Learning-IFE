from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from keras import backend as k
# from model import RecurrentAttentionModel
from tensorflow.examples.tutorials.mnist import input_data
import logging
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
import os
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, concatenate, Activation, BatchNormalization, Input, \
    GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.metrics import confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True  # 按需分配显存，这个比较重要
session = tf.Session(config=config)

sess = tf.Session()
k.set_session(sess)

acc_best=0

def BasicBlock(inpt, filter_num, stride, training):
    x1 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same')(inpt)
    x1 = BatchNormalization()(x1, training=training)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x2 = BatchNormalization()(x2, training=training)

    if stride != 1:
        identify = Conv2D(filter_num, (1, 1), strides=stride)(inpt)
    else:
        identify = inpt

    out = keras.layers.add([x2, identify])
    out = Activation('relu')(out)
    return out

def _weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)


def _bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def _log_likelihood(loc_means, locs, variance):
    loc_means = tf.stack(loc_means)  # [timesteps, batch_sz, loc_dim]
    loc_means = tf.reshape(loc_means, [tf.shape(loc_means)[0], tf.shape(loc_means)[1]])
    locs = tf.stack(locs)
    locs = tf.reshape(locs, [tf.shape(locs)[0], tf.shape(locs)[1]])
    gaussian = Normal(loc_means, variance)
    logll = gaussian._log_prob(x=locs)  # [timesteps, batch_sz, loc_dim]

    # logll = tf.reduce_sum(logll, 2)
    return tf.transpose(logll)  # [batch_sz, timesteps]


class RetinaSensor(object):
    # one scale
    def __init__(self, img_size, pth_size):
        self.img_size = img_size
        self.pth_size = pth_size

    def __call__(self, img_ph, loc):
        img = tf.reshape(img_ph, [tf.shape(img_ph)[0], self.img_size, self.img_size, 1])
        # 5个泳道的中心x坐标，G泳道-0.8，A泳道-0.4，M泳道0.0，k泳道0.4，L泳道0.8
        loc_y1 = -0.8 * tf.cast(tf.random_uniform((tf.shape(img_ph)[0], 1), minval=1, maxval=2, dtype=tf.int32),
                                dtype=tf.float32)
        loc_y2 = -0.4 * tf.cast(tf.random_uniform((tf.shape(img_ph)[0], 1), minval=1, maxval=2, dtype=tf.int32),
                                dtype=tf.float32)
        loc_y3 = 0.0 * tf.cast(tf.random_uniform((tf.shape(img_ph)[0], 1), minval=1, maxval=2, dtype=tf.int32),
                               dtype=tf.float32)
        loc_y4 = 0.4 * tf.cast(tf.random_uniform((tf.shape(img_ph)[0], 1), minval=1, maxval=2, dtype=tf.int32),
                               dtype=tf.float32)
        loc_y5 = 0.8 * tf.cast(tf.random_uniform((tf.shape(img_ph)[0], 1), minval=1, maxval=2, dtype=tf.int32),
                               dtype=tf.float32)

        loc = tf.reshape(loc, [tf.shape(loc)[0], 1])
        loc_y1 = tf.reshape(loc_y1, [tf.shape(loc_y1)[0], 1])
        loc_y2 = tf.reshape(loc_y2, [tf.shape(loc_y2)[0], 1])
        loc_y3 = tf.reshape(loc_y3, [tf.shape(loc_y3)[0], 1])
        loc_y4 = tf.reshape(loc_y4, [tf.shape(loc_y4)[0], 1])
        loc_y5 = tf.reshape(loc_y5, [tf.shape(loc_y5)[0], 1])

        loc_y1 = tf.stop_gradient(loc_y1);
        loc_y2 = tf.stop_gradient(loc_y2)
        loc_y3 = tf.stop_gradient(loc_y3);
        loc_y4 = tf.stop_gradient(loc_y4)
        loc_y5 = tf.stop_gradient(loc_y5)

        loc1 = tf.concat((loc, loc_y1), axis=1)
        loc2 = tf.concat((loc, loc_y2), axis=1)
        loc3 = tf.concat((loc, loc_y3), axis=1)
        loc4 = tf.concat((loc, loc_y4), axis=1)
        loc5 = tf.concat((loc, loc_y5), axis=1)

        loc1 = tf.stop_gradient(loc1);
        loc2 = tf.stop_gradient(loc2)
        loc3 = tf.stop_gradient(loc3);
        loc4 = tf.stop_gradient(loc4)
        loc5 = tf.stop_gradient(loc5)

        pth1 = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc1)
        pth2 = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc2)
        pth3 = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc3)
        pth4 = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc4)
        pth5 = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc5)
        return pth1, pth2, pth3, pth4, pth5

class GlimpseNetwork(object):
    def __init__(self, img_size, pth_size, loc_dim, g_size, l_size, output_size):
        self.retina_sensor = RetinaSensor(img_size, pth_size)

        self.l1_w = _weight_variable((loc_dim, l_size))
        self.l1_b = _bias_variable((l_size,))

        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, imgs_ph, locs, training):
        pths1, pths2, pths3, pths4, pths5 = self.retina_sensor(imgs_ph, locs)
        pths = tf.concat((pths1, pths2, pths3, pths4, pths5), axis=2)

        x_stem = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(pths)  # kernel_size=7, filter=64, stride=2, padding=3(same)
        x_stem = BatchNormalization()(x_stem, training = training)
        x_stem = Activation('relu')(x_stem)
        x_stem = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(
            x_stem)  # kernel_size=3, filter=64, stride=2, padding=3(same)
        # 第一(stride=1)和二(stride=1)个基本模块-64 channel
        x_layer1 = BasicBlock(x_stem, 64, 1, training)
        x_layer1 = BasicBlock(x_layer1, 64, 1, training)
        # 第三(stride=2)和四(stride=1)个基本模块-128 channel
        x_layer2 = BasicBlock(x_layer1, 128, 2, training)
        x_layer2 = BasicBlock(x_layer2, 128, 1, training)
        # 第五(stride=2)和六(stride=1)个基本模块-256 channel
        x_layer3 = BasicBlock(x_layer2, 256, 2, training)
        x_layer3 = BasicBlock(x_layer3, 256, 1, training)
        # 第七(stride=2)和八(stride=1)个基本模块-512 channel
        x_layer4 = BasicBlock(x_layer3, 512, 2, training)
        x_layer4 = BasicBlock(x_layer4, 512, 1, training)
        # 平均池化层
        x_avgpool = GlobalAveragePooling2D()(x_layer4)
        # 全连接输出层
        g = Dense(units=256, activation=None)(x_avgpool)

        locs = tf.reshape(locs, [tf.shape(locs)[0], 1])
        l = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(locs, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)
        return tf.nn.relu(g + l)

class LocationNetwork(object):
    def __init__(self, loc_dim, rnn_output_size, variance=0.22, is_sampling=False):
        self.loc_dim = loc_dim
        self.variance = variance

        self.w = _weight_variable((105, rnn_output_size))
        self.b = _bias_variable((rnn_output_size,))
        self.is_sampling = is_sampling
        self.w1 = _weight_variable((rnn_output_size, loc_dim))
        self.b1 = _bias_variable((loc_dim,))

    def __call__(self, cell_output, img_coa):

        img_coa_trans = tf.nn.xw_plus_b(img_coa, self.w, self.b)
        mean = tf.nn.xw_plus_b(cell_output+img_coa_trans, self.w1, self.b1)
        mean = tf.clip_by_value(mean, -1., 1.)
        mean = tf.stop_gradient(mean)

        if self.is_sampling:
            loc = mean + tf.random_normal(
                (tf.shape(cell_output)[0], self.loc_dim),
                stddev=self.variance)
            loc = tf.clip_by_value(loc[:, 0], -1., 1.)
        else:
            loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean

class RecurrentAttentionModel(object):
    def __init__(self, img_size, pth_size, g_size, l_size, glimpse_output_size,
                 loc_dim, variance,
                 cell_size, num_glimpses, num_classes,
                 learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch,
                 max_gradient_norm,
                 is_training=False):

        self.img_ph = tf.placeholder(tf.float32, [None, img_size * img_size])
        self.lbl_ph = tf.placeholder(tf.int64, [None])
        # self.training_bn = tf.placeholder_with_default(False, [None])
        self.training = tf.placeholder(tf.bool)
        self.img_coa = tf.placeholder(tf.float32, [None, img_size])

        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            training_steps_per_epoch,
            learning_rate_decay_factor,
            staircase=True),
            min_learning_rate)

        cell = BasicLSTMCell(cell_size)

        with tf.variable_scope('GlimpseNetwork'):
            glimpse_network = GlimpseNetwork(img_size, pth_size, loc_dim, g_size, l_size, glimpse_output_size)
        with tf.variable_scope('LocationNetwork'):
            location_network = LocationNetwork(loc_dim=loc_dim, rnn_output_size=cell.output_size, variance=variance,
                                               is_sampling=is_training)

        # Core Network
        batch_size = tf.shape(self.img_ph)[0]
        init_loc = tf.random_uniform((batch_size, 1), minval=-1, maxval=1)
        init_state = cell.zero_state(batch_size, tf.float32)

        init_glimpse = glimpse_network(self.img_ph, init_loc, self.training)
        rnn_inputs = [init_glimpse]
        rnn_inputs.extend([0] * num_glimpses)

        locs, loc_means = [], []

        def loop_function(prev, _):
            loc, loc_mean = location_network(prev, self.img_coa)
            locs.append(loc)
            # print(locs)
            loc_means.append(loc_mean)
            glimpse = glimpse_network(self.img_ph, loc, self.training)
            return glimpse

        rnn_outputs, _ = rnn_decoder(rnn_inputs, init_state, cell, loop_function=loop_function)

        # Time independent baselines
        with tf.variable_scope('Baseline'):
            baseline_w = _weight_variable((cell.output_size, 1))
            baseline_b = _bias_variable((1,))
        baselines = []
        for output in rnn_outputs[1:]:
            baseline = tf.nn.xw_plus_b(output, baseline_w, baseline_b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)
        baselines = tf.stack(baselines)  # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

        # Classification. Take the last step only.
        rnn_last_output = rnn_outputs[-1]
        with tf.variable_scope('Classification'):
            logit_w1 = _weight_variable((cell.output_size, cell.output_size))
            logit_b1 = _weight_variable((cell.output_size,))
            logit_w = _weight_variable((cell.output_size, num_classes))
            logit_b = _bias_variable((num_classes,))
        # logits = tf.nn.xw_plus_b(rnn_last_output, logit_w, logit_b)
        logits = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(rnn_last_output, logit_w1, logit_b1)), logit_w, logit_b)
        self.prediction = tf.argmax(logits, 1)
        self.softmax = tf.nn.softmax(logits)

        if is_training:
            # classification loss
            self.xent = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))
            # RL reward
            reward = tf.cast(tf.equal(self.prediction, self.lbl_ph), tf.float32)
            rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
            rewards = tf.tile(rewards, (1, num_glimpses))  # [batch_sz, timesteps]
            advantages = rewards - tf.stop_gradient(baselines)
            self.advantage = tf.reduce_mean(advantages)
            self.locA = locs
            logll = _log_likelihood(loc_means, locs, variance)
            logllratio = tf.reduce_mean(logll * advantages)
            self.reward = tf.reduce_mean(reward)
            # baseline loss
            self.baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
            # hybrid loss
            self.loss = -logllratio + self.xent + self.baselines_mse
            params = tf.trainable_variables()
            print('params:', params)
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)


logging.getLogger().setLevel(logging.INFO)

x_class9 = np.load(r'.../traindata.npy')
y_class9 = np.load(r'.../trainlabel.npy')
x_sp = np.load(r'.../trainsp.npy')
x_class9 = x_class9.astype('float32'); x_sp = x_sp.astype('float32')
x_sp =np.mean(x_sp, axis=2)
print(x_class9.shape); print(y_class9.shape); print(x_sp.shape)
x_sp = x_sp.reshape(x_sp.shape[0], 105, 1)
x_all = np.concatenate([x_class9, x_sp], axis=2)
print(x_all.shape)
x_train_all, x_valid_all, y_train, y_valid = train_test_split(x_all, y_class9, test_size=0.1, random_state=8, stratify=y_class9)
x_train_all = x_train_all / 255; x_valid_all = x_valid_all / 255
y_train = y_train.reshape(y_train.shape[0], ); y_valid = y_valid.reshape(y_valid.shape[0], )

test_x = np.load(r'.../testdata.npy')
test_y = np.load(r'.../testlabel.npy')
test_sp = np.load(r'.../testsp.npy')
test_x = test_x.astype('float32'); test_sp = test_sp.astype('float32')
test_sp = np.mean(test_sp, axis=2)
test_sp = test_sp.reshape(test_sp.shape[0], 105, 1)
test_x_all = np.concatenate([test_x, test_sp], axis=2)
#test_x = test_x.reshape(test_x.shape[0], 105 * 105)
test_x_all = test_x_all / 255
test_y = test_y.reshape(test_y.shape[0], )
np.random.seed(9)  # 设置随机种子

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 41, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")

tf.app.flags.DEFINE_integer("patch_window_size", 21, "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 256, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size", 256, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("cell_size", 512, "Size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 6, "Number of glimpses.")
tf.app.flags.DEFINE_float("variance", 0.22, "Gaussian variance for Location Network.")
tf.app.flags.DEFINE_integer("M", 10, "Monte Carlo sampling, see Eq(2).")

FLAGS = tf.app.flags.FLAGS


class Dataset:
    def __init__(self, data, label):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    def next_batch(self, batch_size, shuffle=False):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
            self._label = self.label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            label_rest_part = self.label[start:self._num_examples]

            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._label = self.label[idx0]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            label_new_part = self._label[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate(
                (label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._label[start:end]


data_train = Dataset(x_train_all, y_train)
# print(data_train.shape)
data_validation = Dataset(x_valid_all, y_valid)
data_test = Dataset(test_x_all, test_y)

training_steps_per_epoch = x_train_all.shape[0] // FLAGS.batch_size
print(x_train_all.shape[0])
print(training_steps_per_epoch)

ram = RecurrentAttentionModel(img_size=105,  # MNIST: 28 * 28
                              pth_size=FLAGS.patch_window_size,
                              g_size=FLAGS.g_size,
                              l_size=FLAGS.l_size,
                              glimpse_output_size=FLAGS.glimpse_output_size,
                              loc_dim=1,  # (x,y)
                              variance=FLAGS.variance,
                              cell_size=FLAGS.cell_size,
                              num_glimpses=FLAGS.num_glimpses,
                              num_classes=8,
                              learning_rate=FLAGS.learning_rate,
                              learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                              min_learning_rate=FLAGS.min_learning_rate,
                              training_steps_per_epoch=training_steps_per_epoch,
                              max_gradient_norm=FLAGS.max_gradient_norm,
                              is_training=True)
with sess.as_default():
#with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(FLAGS.num_steps):
        images, labels = data_train.next_batch(FLAGS.batch_size)
        images_train = images[:,:,0:105]; images_traincoa = images[:,:,105]
        images_train = images_train.reshape(images_train.shape[0], 105*105)
        images_traincoa = images_traincoa.reshape(images_traincoa.shape[0], 105*1)
        images_train = np.tile(images_train, [FLAGS.M, 1])
        images_traincoa = np.tile(images_traincoa, [FLAGS.M, 1])
        labels = np.tile(labels, [FLAGS.M])
        output_feed = [ram.train_op, ram.loss, ram.xent, ram.reward, ram.advantage, ram.baselines_mse,
                       ram.learning_rate, ram.locA]
        _, loss, xent, reward, advantage, baselines_mse, learning_rate, locA = sess.run(output_feed, \
                                                                                        feed_dict={ram.img_ph: images_train,
                                                                                                   ram.lbl_ph: labels,
                                                                                                   ram.training: True,
                                                                                                   ram.img_coa: images_traincoa})
        # print(locs)
        if step and step % 100 == 0:
            logging.info(
                'step {}: lr = {:3.6f}\tloss = {:3.4f}\txent = {:3.4f}\treward = {:3.4f}\tadvantage = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
                    step, learning_rate, loss, xent, reward, advantage, baselines_mse))
            # print(locA)

        # Evaluation
        if step and step % training_steps_per_epoch == 0:
            for dataset in [data_validation, data_test]:
                if dataset == data_validation:
                    steps_per_epoch = x_valid_all.shape[0] // FLAGS.batch_size
                else:
                    steps_per_epoch = test_x.shape[0] // FLAGS.batch_size
                    pre_mid = []; labels_bak_mid = []
                correct_cnt = 0
                num_samples = steps_per_epoch * FLAGS.batch_size

                for test_step in range(steps_per_epoch):
                    images, labels = dataset.next_batch(FLAGS.batch_size)
                    labels_bak = labels
                    # Duplicate M times
                    images_test = images[:,:,0:105]; images_testcoa = images[:,:,105]
                    images_test = images_test.reshape(images_test.shape[0], 105 * 105)
                    images_testcoa = images_testcoa.reshape(images_testcoa.shape[0], 105 * 1)
                    images_test = np.tile(images_test, [FLAGS.M, 1])
                    images_testcoa = np.tile(images_testcoa, [FLAGS.M, 1])
                    labels = np.tile(labels, [FLAGS.M])
                    # k.set_learning_phase(1)
                    softmax = sess.run(ram.softmax,
                                       feed_dict={ram.img_ph: images_test, ram.lbl_ph: labels, ram.training: True, ram.img_coa: images_testcoa})
                    softmax = np.reshape(softmax, [FLAGS.M, -1, 8])
                    softmax = np.mean(softmax, 0)
                    prediction = np.argmax(softmax, 1).flatten()
                    correct_cnt += np.sum(prediction == labels_bak)
                    if dataset == data_test:
                        pre_mid.append(prediction); labels_bak_mid.append(labels_bak)
                    # print(prediction); print(correct_cnt)
                acc = correct_cnt / num_samples
                if dataset == data_test:
                    if acc>acc_best:
                        acc_best = acc
                        ram.saver.save(sess, ".../ODRAM_best.ckpt")
                        pre_mid = np.array(pre_mid).reshape(-1,1); labels_bak_mid = np.array(labels_bak_mid).reshape(-1,1)
                        i_ak = 0; i_aL = 0; i_gk = 0; i_gL = 0; i_mk = 0; i_mL = 0; i_k = 0; i_L = 0
                        for ui in range(labels_bak_mid.shape[0]):
                            if labels_bak_mid[ui] == 0: i_ak = i_ak + 1
                            elif labels_bak_mid[ui] == 1: i_aL = i_aL + 1
                            elif labels_bak_mid[ui] == 2: i_gk = i_gk + 1
                            elif labels_bak_mid[ui] == 3: i_gL = i_gL + 1
                            elif labels_bak_mid[ui] == 4: i_mk = i_mk + 1
                            elif labels_bak_mid[ui] == 5: i_mL = i_mL + 1
                            elif labels_bak_mid[ui] == 6: i_k = i_k + 1
                            elif labels_bak_mid[ui] == 7: i_L = i_L + 1
                        print('i_ak=', i_ak); print('i_aL=', i_aL); print('i_gk=', i_gk); print('i_gL=', i_gL)
                        print('i_mk=', i_mk); print('i_mL=', i_mL); print('i_k=', i_k); print('i_L=', i_L)
                        C2 = confusion_matrix(labels_bak_mid, pre_mid, labels=[0, 1, 2, 3, 4, 5, 6, 7])
                        print(C2)
                if dataset == data_validation:
                    logging.info('valid accuracy = {}'.format(acc))
                else:
                    logging.info('test accuracy = {}'.format(acc))