from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import torchvision
import os
import math
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
import PIL.Image as Image
import skimage.io as io
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
import seaborn as sns
from keras.models import load_model
import xlrd

from keras.utils import np_utils
import tensorflow as tf
import keras
from keras import backend as k
from tensorflow.examples.tutorials.mnist import input_data
import logging
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class IFEDataset(torch.utils.data.Dataset):
    def __init__(self, x, label, transform=None):
        self.x = x
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data_x = torch.from_numpy(self.x[idx]).type(torch.FloatTensor)
        data_y = torch.from_numpy(np.array(self.label[idx])).type(torch.long)
        return data_x, data_y


args = {"device": "cuda:0", "total_epoch": 40, "mile_stones": [10, 30], "lr": 0.1, "optim": 'sgd', "attn_type": None,  # use CBAM
    "diagonal_input": True, "use_cag": [True] * 4,  # use CAG
    "model_save_dir": ".../final_model/", "img_info_path": ".../test_csv.csv",
    "data_path": ".../data_pnclass_tensor.npy",
    "img_path": ".../imags/",
    "blob_path": ".../imag_blobs/"}

device = torch.device(args["device"])
total_label = []
total_predictions = []
file = pd.read_csv(args["img_info_path"])
Y_data = file["MULTI-RESULT"].values
fold = file["FOLD"].values

if args["diagonal_input"]:
    X_data = np.load(args["data_path"])
    X_data = X_data * get_mask(X_data.shape[-1])
else:
    X_data = np.load(args["data_path"])

i_fold=0
torch.cuda.empty_cache()
train_index = fold != i_fold; val_index = fold == i_fold

best_val_loss = 100.; best_val_score = 0.; best_epoch_pred = []
train_loss_list = []; val_loss_list = []; train_acc_list = []

val_acc_list = []; train_f1_list = []; val_f1_list = []

# prepare train data and val data
x_train = X_data[train_index]  # train data
x_val = X_data[val_index]
y_train = Y_data[train_index]  # label
y_val = Y_data[val_index]

# dataset defination
train_dataset = IFEDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_dataset = IFEDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

# model
model = ResidualNet(network_type='ImageNet', depth=18, num_classes=2, att_type=args["attn_type"],use_mask=args["use_cag"])
model.load_state_dict(torch.load('.../dcl_pnclass_0.pkl'))
model = model.to(device)
criterion = nn.CrossEntropyLoss()

epoch_label = np.array(y_val).tolist()
running_loss = 0.
val_running_loss = 0.
epoch_pred = []
train_label = []
train_pred = []
# ************** start to validate *****************
model.eval()
for x, y in val_loader:
     x = x.to(device)
     y = y.to(device)
     prob = model(x)
     _, pred = torch.max(prob, 1)
     val_loss = criterion(prob, y)
     val_running_loss += val_loss.item() * x.shape[0]
     epoch_pred.extend(pred.data.cpu().numpy().tolist())
# update evalution criteria
print('epoch_pred=',epoch_pred)
print('epocl_labe=',epoch_label)
epoch_pred_ar = np.array(epoch_pred)
epoch_label_ar = np.array(epoch_label)
false_sam = []; false_sam_ptn = []; clas_pos = []
for ik in range(epoch_pred_ar.shape[0]):
    if epoch_pred_ar[ik] != epoch_label_ar[ik]:
        false_sam.append(ik+1)
    if epoch_pred_ar[ik] == 0 and epoch_label_ar[ik] == 1:
        false_sam_ptn.append(ik+1)
    if epoch_pred_ar[ik] == 1:
        clas_pos.append(ik+1)
print(false_sam); print('false_sam.shape', np.array(false_sam).shape)
print(false_sam_ptn); print('false_sam_ptn.shape', np.array(false_sam_ptn).shape)
print(clas_pos); print('clas_pos.shape', np.array(clas_pos).shape)

val_epoch_loss = val_running_loss / len(y_val)
epoch_acc = accuracy_score(epoch_label, epoch_pred)
epoch_f1 = f1_score(epoch_label, epoch_pred, average='macro')
print('epoch_f1=',epoch_f1)
# update list
val_loss_list.append(val_epoch_loss)
val_acc_list.append(epoch_acc)
val_f1_list.append(epoch_f1)
# print
print('val loss : {:.4f}, val acc : {:.4f}, val f1 : {:.4f}'.format(val_epoch_loss, epoch_acc, epoch_f1))

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
k.set_session(sess)

logging.getLogger().setLevel(logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 30, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")
tf.app.flags.DEFINE_integer("patch_window_size", 21, "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 256, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size", 256, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("cell_size", 512, "Size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 12, "Number of glimpses.")
tf.app.flags.DEFINE_float("variance", 0.22, "Gaussian variance for Location Network.")
tf.app.flags.DEFINE_integer("M", 10, "Monte Carlo sampling, see Eq(2).")
FLAGS = tf.app.flags.FLAGS

train_x2 = np.load(r'.../testdata_pos.npy')
label_y2 = np.load(r'.../testlabel_pos.npy')

p=0; numb_pos=0
while p<nrows2-1:
    if epoch_pred[p]==1: numb_pos=numb_pos+1
    p=p+1
print(numb_pos)

z=0; zi=0; test_x_pos = np.zeros((numb_pos,105,105))
test_y_label = np.zeros((numb_pos,1))
while z<nrows2-1:
    if epoch_pred[z]==1:    #如果在阴阳分类中分成了阳性
        test_x_pos[zi,:,:] = train_x2[z,:,:]
        test_y_label[zi] = label_y2[z]
        zi=zi+1
    z=z+1

test_x_pos = test_x_pos.astype('float32'); test_x_possp = test_x_possp.astype('float32')
test_x_possp = test_x_possp.reshape(test_x_possp.shape[0], 105, 1)
test_x_all = np.concatenate([test_x_pos, test_x_possp], axis=2)
#test_x = test_x.reshape(test_x.shape[0], 105 * 105)
test_x_all = test_x_all / 255
test_y_label = test_y_label.reshape(test_y_label.shape[0], )

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
            self._data = self.data[idx]  # get list of `num` random samples
            self._label = self.label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            label_rest_part = self.label[start:self._num_examples]

            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._label = self.label[idx0]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            label_new_part = self._label[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate(
                (label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._label[start:end]

data_test = Dataset(test_x_all, test_y_label)
training_steps_per_epoch = 8449 // FLAGS.batch_size
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
    ram.saver.restore(sess, ".../RAM_posclass.ckpt")
    batch_size_test = 1; M1 = 10
    steps_per_epoch = test_x_all.shape[0] // batch_size_test
    pre_mid = []; labels_bak_mid = []
    correct_cnt = 0
    num_samples = steps_per_epoch * batch_size_test

    for test_step in range(steps_per_epoch):
        images, labels = data_test.next_batch(batch_size_test)
        labels_bak = labels

        images_test = images[:, :, 0:105]; images_testcoa = images[:, :, 105]
        images_test = images_test.reshape(images_test.shape[0], 105 * 105)
        images_testcoa = images_testcoa.reshape(images_testcoa.shape[0], 105 * 1)
        images_test = np.tile(images_test, [M1, 1])
        images_testcoa = np.tile(images_testcoa, [M1, 1])
        labels = np.tile(labels, [M1])
        softmax = sess.run(ram.softmax,feed_dict={ram.img_ph: images_test, ram.lbl_ph: labels, ram.training: True,
                                          ram.img_coa: images_testcoa})
        softmax = np.reshape(softmax, [M1, -1, 8])
        softmax = np.mean(softmax, 0)
        prediction = np.argmax(softmax, 1).flatten()
        correct_cnt += np.sum(prediction == labels_bak)
        pre_mid.append(prediction); labels_bak_mid.append(labels_bak)
    acc = correct_cnt / num_samples
    pre_mid_ar = np.array(pre_mid).reshape(-1,)
    labels_bak_mid_ar = np.array(labels_bak_mid).reshape(-1,)

    pre_mid = np.array(pre_mid).reshape(-1, 1)

    labels_bak_mid = np.array(labels_bak_mid).reshape(-1, 1)
    false_sam_pos = []; pre_mid_false = []; lab_mid_false = []
    for jk in range(pre_mid_ar.shape[0]):
        if pre_mid_ar[jk] != int(labels_bak_mid_ar[jk]):
            false_sam_pos.append(clas_pos[jk])
            pre_mid_false.append(pre_mid_ar[jk])
            lab_mid_false.append(int(labels_bak_mid_ar[jk]))
    print('false_sam_pos=', false_sam_pos)
    print('pre_mid_false=', pre_mid_false)
    print('lab_mid_false=', lab_mid_false)
    print('fasle_pos_num=', np.array(lab_mid_false).shape)

    C2 = confusion_matrix(labels_bak_mid, pre_mid, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    print(C2)





