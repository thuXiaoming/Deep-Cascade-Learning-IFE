import warnings
warnings.simplefilter('ignore')
import pandas as pd
import torch
import numpy as np
import torchvision
import os
import torch.nn as nn
import math
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
import PIL.Image as Image
import skimage.io as io
import torch.nn.functional as F
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
from sklearn.metrics import confusion_matrix
#from models.ResNet_CAG import *
#from utils.aux_func import *
#from utils.dataset import IFEDataset

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_mask(size):
    """
    Description:
        Generate a mask which follows normal distribution
    Args:
        size (int): height and width of the target mask
    """
    mask = np.zeros([size, size])
    u = 0  # 均值μ
    sig = math.sqrt(1)  # 标准差δ
    x = np.linspace(0, u + 3 * sig, size)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    for i in range(size):
        for j in range(size):
            mask[i, j] = y_sig[np.abs(i - j)]

    mask = mask / np.max(mask)
    return mask

class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes, att_type=None, use_mask=[False, False, False, False],input_size=105):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        input_size = int(input_size / 2) + input_size % 2
        self.layer1 = self._make_layer(block, 64, layers[0], att_type=att_type, use_mask=use_mask[0],feature_map_size=input_size)
        input_size = int(input_size / 2) + input_size % 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type, use_mask=use_mask[1],feature_map_size=input_size)
        input_size = int(input_size / 2) + input_size % 2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type, use_mask=use_mask[2],feature_map_size=input_size)
        input_size = int(input_size / 2) + input_size % 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type, use_mask=use_mask[3],feature_map_size=input_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None, use_mask=False, feature_map_size=105):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type == 'CBAM', use_mask=use_mask,
                            feature_map_size=feature_map_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type == 'CBAM', use_mask=use_mask,
                                feature_map_size=feature_map_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class CAG_Module(nn.Module):
    def __init__(self, feature_map_size):
        super(CAG_Module, self).__init__()
        self.weight = nn.Parameter(data=torch.FloatTensor([1., 1.]), requires_grad=True)
        self.coordinate = nn.Parameter(data=self.get_x(int(feature_map_size)), requires_grad=False)
        self.pi = nn.Parameter(data=torch.tensor([2 * np.pi]), requires_grad=False)
    def get_x(self, size):
        a = np.linspace(1, 4, size)
        y = np.flip(a, axis=0)
        a = np.concatenate((y[:-1], a))
        b = np.array([a[size - 1 - i: 2 * size - 1 - i] for i in range(size)])
        coordinate = torch.from_numpy(b).float()
        return coordinate
    def forward(self, x):
        y_sig = torch.exp(-(self.coordinate - self.weight[0]) ** 2 / (2 * self.weight[1] ** 2)) / (
                    torch.sqrt(self.pi) * self.weight[1])
        diagonal_output = torch.mul(x, y_sig)
        out = x + diagonal_output
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, use_mask=False,feature_map_size=100):
        super(BasicBlock, self).__init__()
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if use_mask:
            self.cag = CAG_Module(feature_map_size)
        else:
            self.cag = None
        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if not self.cag is None:
            out = self.cag(out)
        if not self.cbam is None:
            out = self.cbam(out)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)
        return out

def ResidualNet(network_type, depth, num_classes, att_type, use_mask=[False, False, False, False], input_size=52):
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'
    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type, use_mask, input_size)
    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type, use_mask, input_size)
    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type, use_mask, input_size)
    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type, use_mask, input_size)
    return model

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

def draw_loss(total_epoch, train_loss_list, val_loss_list, dynamic_y=False):
    """
    Args:
        total_epoch (int): iter number of training
        train_loss_list (list/np.array): train loss
        val_loss_list (list/np.array): val loss
        dynamic_y (bool): change the y_max according to the loss
    return:
        None
    """
    plt.figure()
    if dynamic_y:
        y_max = max(max(val_loss_list), max(train_loss_list))
    else:
        y_max = 2
    x = np.linspace(0, total_epoch, num=total_epoch)
    plt.plot(x, train_loss_list, color='r', label='train_loss')
    plt.plot(x, val_loss_list, 'b', label='val_loss')#'b'指：color='blue'
    plt.legend()  #显示上面的label
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axis([0, total_epoch, 0, y_max])#设置坐标范围axis([xmin,xmax,ymin,ymax])
    plt.show()

def draw_acc(total_epoch, train_acc, val_acc, ylabel):
    """
    Args:
        total_epoch (int): iter number of training
        train_acc (list/np.array): train accuracy (no matter what kind accuracy)
        val_acc (list/np.array): val accuracy
        ylabel (string): the y label text
    return:
        None
    """
    plt.figure()
    x = np.linspace(0, total_epoch, num=total_epoch)
    plt.plot(x, train_acc, color='r', label='train_'+ylabel)
    plt.plot(x, val_acc, 'b', label='val_'+ylabel)#'b'指：color='blue'
    plt.legend()  #显示上面的label
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.axis([0, total_epoch, 0.7, 1.0])#设置坐标范围axis([xmin,xmax,ymin,ymax])
    plt.show()

def train(args):
    device = torch.device(args["device"])

    if not os.path.isdir(args["model_save_dir"]):
        os.makedirs(args["model_save_dir"])

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

    for i_fold in range(1):
        model_save_path = '{}dcl_pnclass_{}.pkl'.format(args["model_save_dir"], i_fold)
        torch.cuda.empty_cache()
        train_index = fold != i_fold
        val_index = fold == i_fold

        print('********************************************************************')
        print('****************************{} fold start***************************'.format(i_fold))
        print('********************************************************************')

        best_val_loss = 100.
        best_val_score = 0.
        best_epoch_pred = []

        train_loss_list = []
        val_loss_list = []
        train_acc_list = []

        val_acc_list = []
        train_f1_list = []
        val_f1_list = []

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
        model = ResidualNet(network_type='ImageNet', depth=18, num_classes=2, att_type=args["attn_type"],
                            use_mask=args["use_cag"])
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        if args["optim"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args["mile_stones"], gamma=0.1)

        epoch_label = np.array(y_val).tolist()
        for epoch in range(args["total_epoch"]):
            # 每次迭代的评价指标
            running_loss = 0.
            val_running_loss = 0.
            epoch_pred = []
            train_label = []
            train_pred = []

            print("*************************************************************")
            print("{} epoch train start:".format(epoch))
            print(scheduler.get_lr())

            # ************** start to train ********************
            model.train()

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(x)
                _, pred = torch.max(outputs, 1)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.shape[0]
                train_label.extend(y.data.cpu().numpy().tolist())
                train_pred.extend(pred.data.cpu().numpy().tolist())

            scheduler.step()
            # update evaluation criteria
            epoch_loss = running_loss / len(y_train)
            train_acc = accuracy_score(train_label, train_pred)
            train_f1 = f1_score(train_label, train_pred, average='macro')
            # update list
            train_loss_list.append(epoch_loss)
            train_f1_list.append(train_f1)
            train_acc_list.append(train_acc)
            # print
            print('train loss : {:.4f}, train acc : {:.4f}, train f1 : {:.4f}'.format(epoch_loss, train_acc, train_f1))

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

            # *************** save the best model ************************
            if epoch_f1 >= best_val_score:
                best_val_score = epoch_f1
                print('best score now !!!!!!!!!!!!!!************!!!!!!!!!'.format(epoch))
                torch.save(model.state_dict(), model_save_path)
                best_epoch_pred = epoch_pred
        print('epoch_label=',epoch_label); print('best_epoch_pred=',best_epoch_pred)
        # draw loss and print metrics every fold
        print(classification_report(epoch_label, best_epoch_pred, digits=4))
        draw_loss(args["total_epoch"], train_loss_list, val_loss_list, True)
        draw_acc(args["total_epoch"], train_acc_list, val_acc_list, 'acc')
        draw_acc(args["total_epoch"], train_f1_list, val_f1_list, 'f1')

        # update total fold matrix
        total_label.extend(epoch_label)
        total_predictions.extend(best_epoch_pred)
    # print total fold metrics
    #print(classification_report(total_label, total_predictions, digits=4))

    sns.set()
    C2 = confusion_matrix(epoch_label, best_epoch_pred, labels=[0, 1])
    print(C2)  # 打印出来看看
    # sns.heatmap(C2,annot=True,ax=ax) #画热力图
    sns.heatmap(C2, annot=True, fmt="d")  # 画热力图cmap="YlGnBu"

    plt.title('confusion matrix')
    plt.xlabel('predict')
    plt.ylabel('true')
    plt.show()

args = {"device": "cuda:0", "total_epoch": 40, "mile_stones": [10, 30], "lr": 0.1, "optim": 'sgd', "attn_type": None,  # use CBAM
    "diagonal_input": True, "use_cag": [True] * 4,  # use CAG
    "model_save_dir": ".../final_model/", "img_info_path": ".../test_csv.csv",
    "data_path": ".../data_pnclass_tensor.npy",
    "img_path": ".../imags/",
    "blob_path": ".../imag_blobs/"}
train(args)


