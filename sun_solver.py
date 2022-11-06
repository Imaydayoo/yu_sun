# -*- coding: UTF-8 -*-
import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim, Tensor
from torch.autograd import Variable
import torch.nn.functional as F

from evaluation import *
from model.MSSN_GN.models import MyUnet_GN
from model.MSSN_GN_1c.models import MyUnet_GN_1C
from model.My_Enet.My_ENet import My_ENet

from model.aspp_unet.aspp_unet import ASPP_U_Net
from model.aspp_unet.aspp_unet_4 import ASPP_U_Net_4
from model.aspp_unet.replace_aspp_unet import Replace_ASPP_U_Net_4
# from model.dcsaunet.DCSAU_Net import DCSAU
# from model.dcsaunet.DCSAU_Net import DCSAU
from model.deeplabv3_light.deeplabv3_model import deeplabv3_resnet50_light
from model.fast_scnn.fast_scnn import FastSCNN
from model.light.my_light import My_light

from model.lraspp import lraspp_mobilenetv3_large
from model.multiresunet.multiresunet import MultiResUnet
from model.multiresunet.my_135_unet import My_135_U_Net
from model.multiresunet.my_res_133312_unet import My_res_133312_U_Net
from model.multiresunet.my_res_1333_unet import My_res_1333_U_Net
from model.multiresunet.my_res_133_unet import My_res_133_U_Net
from model.pranet.PraNet_Res2Net import PraNet
from model.deeplabv3.deeplabv3_model import deeplabv3_resnet50
from model.resunet_pkg.res_unet import ResUnet
from model.saunet.models import SAUNet
from model.unet_my.loss import DualLoss_SAU
from model.sun.unet_multi import Multi_U_Net
from model.u2net.u2_model import u2net_full
from model.unet.cbam_unet import CBAM_U_Net
from model.unet_my.models import MyUnet
from model.unet_shape.loss import DualLoss_Shape
from model.unet_shape.models import UnetShape
from model.unet_shape_mul_aspp.models import MyUnet_Aspp
from model.unet_shape_mul_at.models import MyUnet_At
from model.unet_shape_mul_attu.models import MyUnet_ATTU
from model.unet_shape_mul_deform.models import MyUnet_Deform
from model.unet_shape_mul_res.models import MyUnet_Res
from model.unetx.unetx import UNetX
from network import FCN, U_Net, R2U_Net, AttU_Net, R2AttU_Net, V_Net, MDV_Net, BM_U_Net, init_weights, MD_UNet, \
    U_YS_Net, U_YS_Net_64, U_YS_Net_16, MBU_YS_Net_16, RealU_YS_Net_16, DBU_YS_Net_16, MDBU_YS_Net_16, AG_DBU_YS_Net_16, SB_DBU_YS_Net_16, AG_BU_YS_Net_16
from ENet import ENet
from ERFNet import ERFNet
from GACN import GACN
from BiseNetV2 import BiseNetV2
from ellipse import drawline_AOD
from ellipse2 import drawline_AOD as drawline_AOD2
import csv
import cv2 as cv
import cv2
# from ranger import Ranger
from loss_func import FocalLoss, FocalLoss2d, BinaryDiceLoss, DiceLoss, Mul_FocalLoss, Convey_Loss, Shape_prior_Loss, u2net_loss
# from DeformConvnet import Deform_U_Net,MD_Defm_V_Net
from tensorboardX import SummaryWriter  # Tensorboard显示
# import torch.onnx.symbolic_opset9
from thop import profile
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from sun_fcn import fcn_resnet50
from ys_ellipse import get_endpoint, get_tangent_point
import math
from data_loader import get_loader
import surface_distance
from torchvision import transforms as T
# @torch.onnx.symbolic_opset9.parse_args('v', 'is')
# def upsample_nearest2d(g, input, output_size):
# 	height_scale = float(output_size[-2]) / input.type().sizes()[-2]
# 	width_scale = float(output_size[-1]) / input.type().sizes()[-1]
# 	return g.op("Upsample", input,
# 		scales_f=(1, 1, height_scale, width_scale),
# 		mode_s="nearest")
# torch.onnx.symbolic_opset9.upsample_nearest2d = upsample_nearest2d

def getAllTour(GT_Tensor:Tensor) ->Tensor :
    x = GT_Tensor
    x_size = x.size()
    im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
    canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
    for i in range(x_size[0]):
        canny[i] = cv2.Canny(im_arr[i], 10, 100)
    canny = torch.from_numpy(canny).cuda().float()
    return canny


class SunSolver(object):

    def __init__(self, config, train_loader, valid_loader, test_loader):
        if config == None:
            return
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.5, 2.,
                                                                                     0.2])).float())  # self.criterion = torch.nn.BCELoss()--weight=torch.from_numpy(np.array([0.1,2.,0.5])).float()

        self.criterion1 = Mul_FocalLoss()
        self.criterion2 = torch.nn.MSELoss(reduction='mean')
        # size_average = true，返回的是loss.mean()

        # size_average = false，返回的是loss.sum()
        torch.nn.MSELoss()
        self.criterion3 = DiceLoss(ignore_index=0, max_epoch=config.num_epochs)
        self.criterion3_2 = DiceLoss(ignore_index=0, max_epoch=config.num_epochs)
        self.criterion3_3 = DiceLoss(ignore_index=0, max_epoch=config.num_epochs)
        self.criterion3_4 = DiceLoss(ignore_index=0, max_epoch=config.num_epochs)
        self.criterion3_5 = DiceLoss(ignore_index=0, max_epoch=config.num_epochs)
        self.criterionBCE = torch.nn.BCELoss()
        self.criterionDICE = BinaryDiceLoss()
        self.criterion_CE = torch.nn.CrossEntropyLoss()
        # self.criterion4 = lovasz_softmax()
        self.criterion_convey = Convey_Loss()
        self.criterion_shapeprior = Shape_prior_Loss()

        self.dualLoss = DualLoss_SAU()
        self.dualLoss_shape = DualLoss_Shape()

        if torch.cuda.is_available():
            self.criterion = self.criterion.cuda()

            self.criterion1 = self.criterion1.cuda()
            self.criterion2 = self.criterion2.cuda()
            self.criterion3 = self.criterion3.cuda()
            self.criterion3_2 = self.criterion3_2.cuda()
            self.criterion3_3 = self.criterion3_3.cuda()
            self.criterion3_4 = self.criterion3_4.cuda()
            self.criterion3_5 = self.criterion3_5.cuda()
            self.criterionBCE = self.criterionBCE.cuda()
            self.criterionDICE = self.criterionDICE.cuda()
            self.criterion_CE = self.criterion_CE.cuda()
            # self.criterion4 = lovasz_softmax()
            self.criterion_convey = self.criterion_convey.cuda()
            self.criterion_shapeprior = self.criterion_shapeprior.cuda()


        self.augmentation_prob = config.augmentation_prob
        # image size
        self.image_size = config.image_size

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.EM_MOM = config.EM_momentum
        self.EM_iternum = config.EM_iternum

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net' or self.model_type == 'U_Net_YU':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)  # img_ch为输入通道数，原为3
        if self.model_type == 'U_Net4000':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)  # img_ch为输入通道数，原为3
        if self.model_type == 'U_Net_d1' or self.model_type == 'U_Net_nd1':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)  # img_ch为输入通道数，原为3
        if self.model_type == 'U_Net_d2':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)  # img_ch为输入通道数，原为3
        if self.model_type == 'U_Net_d3':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)  # img_ch为输入通道数，原为3
        if self.model_type == 'U_Net_d4':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)  # img_ch为输入通道数，原为3
        if self.model_type == 'U_Net_d5':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)  # img_ch为输入通道数，原为3
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net' or self.model_type == 'AttU_Net_YU':
            self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'V_Net':
            self.unet = V_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'MDV_Net':
            self.unet = MDV_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        # elif self.model_type == 'BM_U_Net':
        #     self.unet = BM_U_Net(img_ch=self.img_ch, output_ch=self.output_ch, iter_num=self.EM_iternum)
        # elif self.model_type == 'Deform_U_Net':
        # 	self.unet = Deform_U_Net(img_ch=self.img_ch, output_ch=self.output_ch, iter_num=self.EM_iternum)
        # elif self.model_type == 'MD_Defm_V_Net':
        # 	self.unet = MD_Defm_V_Net(img_ch=self.img_ch, output_ch=self.output_ch, iter_num=self.EM_iternum)
        elif self.model_type == 'FCN':
            self.unet = FCN(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'MD_UNet':
            self.unet = MD_UNet(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'U_YS_Net':
            self.unet = U_YS_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'U_YS_Net_64':
            self.unet = U_YS_Net_64(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'U_YS_Net_16':
            self.unet = U_YS_Net_16(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'MBU_YS_Net_16':
            self.unet = MBU_YS_Net_16(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'RealU_YS_Net_16':
            self.unet = RealU_YS_Net_16(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'DBU_YS_Net_16':
            self.unet = DBU_YS_Net_16(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'MDBU_YS_Net_16':
            self.unet = MDBU_YS_Net_16(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'AG_DBU_YS_Net_16' or self.model_type == 'AG_DBU_YS_Net_16_YU':
            self.unet = AG_DBU_YS_Net_16(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'SB_DBU_YS_Net_16':
            self.unet = SB_DBU_YS_Net_16(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'AG_BU_YS_Net_16':
            self.unet = AG_BU_YS_Net_16(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'ENet':
            self.unet = ENet(3)
        elif self.model_type == 'ERFNet':
            self.unet = ERFNet(3)
        elif self.model_type == 'GACN':
            self.unet = GACN(3)
        elif self.model_type == 'BiseNetV2':
            self.unet = BiseNetV2(3)
        elif self.model_type == 'fcn_resnet50':
            self.unet = fcn_resnet50(aux=False, num_classes=3)
        elif self.model_type == 'fcn_resnet50_d1':
            self.unet = fcn_resnet50(aux=False, num_classes=3)
        elif self.model_type == 'fcn_resnet50_d2':
            self.unet = fcn_resnet50(aux=False, num_classes=3)
        elif self.model_type == 'fcn_resnet50_d3':
            self.unet = fcn_resnet50(aux=False, num_classes=3)
        elif self.model_type == 'fcn_resnet50_d4':
            self.unet = fcn_resnet50(aux=False, num_classes=3)
        elif self.model_type == 'fcn_resnet50_d5':
            self.unet = fcn_resnet50(aux=False, num_classes=3)
        elif self.model_type == 'deeplabv3_resnet50':
            self.unet = deeplabv3_resnet50(aux=False, num_classes=3)
        elif self.model_type == 'deeplabv3_resnet50d5':
            self.unet = deeplabv3_resnet50(aux=False, num_classes=3)
        elif self.model_type == 'PraNet':
            self.unet = PraNet()
        elif self.model_type == 'lraspp':
            self.unet = lraspp_mobilenetv3_large(num_classes=3)
        # elif self.model_type == 'dscaunet':
        #     self.unet = DCSAU(img_channels=1, n_classes=3)
        elif self.model_type == 'CBAM_U_Net':
            self.unet = CBAM_U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'ASPP_U_Net':
            self.unet = ASPP_U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'Replace_ASPP_U_Net':
            self.unet = Replace_ASPP_U_Net_4(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'multiresunet':
            self.unet = MultiResUnet(channels=1, nclasses=3)
        elif self.model_type == 'My_135_U_Net':
            self.unet = My_135_U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'My_res_133_U_Net':
            self.unet = My_res_133_U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'My_res_1333_U_Net':
            self.unet = My_res_1333_U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'My_res_133312_U_Net':
            self.unet = My_res_133312_U_Net(img_ch=self.img_ch, output_ch=self.output_ch)

        elif self.model_type == 'U2_Net':
            self.unet = u2net_full()
        elif self.model_type == 'Multi_U_Net':
            self.unet = Multi_U_Net()
        elif self.model_type == 'Res_Unet':
            self.unet = ResUnet(channel=1)
        elif self.model_type == 'SAU_Net':
            self.unet = SAUNet(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_no_attention':
            self.unet = MyUnet(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_no_attention_g123':
            self.unet = MyUnet(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_no_attention4000':
            self.unet = MyUnet(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_no_attention_d2':
            self.unet = MyUnet(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_no_attention_d3':
            self.unet = MyUnet(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_no_attention_d4':
            self.unet = MyUnet(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_no_attention_d5': #d12345 区别就在于数据集不同
            self.unet = MyUnet(num_classes=3)
        elif self.model_type == 'My_Unet_no_multi_has_shape_no_attention':
            self.unet = UnetShape(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_has_aspp':
            self.unet = MyUnet_Aspp(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_has_res':
            self.unet = MyUnet_Res(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_has_attu':
            self.unet = MyUnet_ATTU(num_classes=3)
        elif self.model_type == 'My_Unet_has_multi_has_shape_has_deform':
            self.unet = MyUnet_Deform(num_classes=3)
        elif self.model_type == 'MSSN_GN':
            self.unet = MyUnet_GN(num_classes=3)
        elif self.model_type == 'MSSN_GN_1C' or self.model_type == 'MSSN_GN_1C_nd1':
            self.unet = MyUnet_GN_1C(num_classes=3)
        elif self.model_type == 'MD_UNet':
            self.unet = MD_UNet(img_ch=self.img_ch, output_ch=3)

        elif self.model_type == 'UnetX':
            self.unet = UNetX(img_ch=1, output_ch=3)
        elif self.model_type == 'deeplabv3_resnet50_light':
            self.unet = deeplabv3_resnet50_light(aux=False, num_classes=3)
        elif self.model_type == 'My_light':
            self.unet = My_light(num_classes=3)
        elif self.model_type == 'fast_scnn':
            self.unet = FastSCNN(3)
        elif self.model_type == 'My_ENet':
            self.unet = My_ENet(num_classes=3)

        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.unet.parameters()),
        # 							self.lr, [self.beta1, self.beta2])
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)
        self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)  # 大于0.5的判定为分割区域

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def img_catdist_channel(self, img):
        img_w = img.size()[2]
        img_h = img.size()[3]
        dist_channel = torch.tensor([i for i in range(0, img_w)]).repeat(img_h, 1).t().unsqueeze(0).unsqueeze(0).float()
        # print(dist_channel.dtype)
        # print(img.dtype)
        img_cat = torch.cat((img, dist_channel), 1)

        return img_cat

    # 012对应的onehot编码方法
    def onehot_to_mulchannel(self, GT):
        xx = GT.max() + 1
        # print(xx)


        for i in range(GT.max() + 1):
            if i == 0:
                GT_sg = GT == i
            else:
                GT_sg = torch.cat([GT_sg, GT == i], 1)
        # GT_sg_0 = GT == 0
        # GT_sg_1 = GT == 1
        # GT_sg_2 = GT == 2
        # GT_sg = torch.cat([GT_sg_0, GT_sg_2, GT_sg_1], 1)
        return GT_sg.float()

    # 0 76 150 对应的onehot编码方法
    def onehot_to_mulchannel_sun(self, GT):
        xx = GT.max() + 1

        class_color = [0, 76, 150]
        for i in range(3):
            if i == 0:
                GT_sg = GT == class_color[i]
            else:
                GT_sg = torch.cat([GT_sg, GT == class_color[i]], 1)

        return GT_sg.float()




    def gray2color(self, gray_array, color_map):

        rows, cols = gray_array.shape
        color_array = np.zeros((rows, cols, 3), np.uint8)

        for i in range(0, rows):
            for j in range(0, cols):
                color_array[i, j] = color_map[gray_array[i, j]]

        # color_image = Image.fromarray(color_array)

        return color_array

    def plot_metric(self, history_train, history_valid, metric):
        # history_train = pd.read_csv("./result/csv/train_history.csv")
        # history_valid = pd.read_csv("./result/csv/valid_history.csv")
        train_metrics = history_train[metric]
        valid_metrics = history_valid[metric]
        # train_metrics = dfhistory[metric]
        # val_metrics = dfhistory['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'bo--')
        plt.plot(epochs, valid_metrics, 'ro-')
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.show()

    def plot_metric_single(self, history_train, metric):
        # history_train = pd.read_csv("./result/csv/train_history.csv")
        # history_valid = pd.read_csv("./result/csv/valid_history.csv")
        train_metrics = history_train[metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'bo--')
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric])
        plt.show()

    def show_history(self):
        history_train = pd.read_csv("./result/csv/train_history.csv")
        history_valid = pd.read_csv("./result/csv/valid_history.csv")
        self.plot_metric_single(history_train, 'loss')
        self.plot_metric(history_train, history_valid, 'acc')
        self.plot_metric(history_train, history_valid, 'SE')
        # self.plot_metric(history_train, history_valid, 'SP')
        self.plot_metric(history_train, history_valid, 'PC')
        self.plot_metric(history_train, history_valid, 'F1')
        self.plot_metric(history_train, history_valid, 'JS')
        self.plot_metric(history_train, history_valid, 'DC')
        self.plot_metric(history_train, history_valid, 'DC1')
        self.plot_metric(history_train, history_valid, 'DC2')
        print()

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        time_begin_second = time.time()
        time_begin = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_begin_second))

        dfhistory_train = pd.DataFrame(
            columns=["epoch", "cost_time", "loss", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "DC1", "DC2"])
        dfhistory_valid = pd.DataFrame(
            columns=["epoch", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "DC1", "DC2", "angle_div"])
        dfhistory_test = pd.DataFrame(
            columns=["epoch", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "DC1", "DC2"])
        # dfhistory_train = pd.read_csv('./result/csv/train_history.csv')
        # dfhistory_valid = pd.read_csv('./result/csv/valid_history.csv')
        print('=' * 50)
        print("Train start at:" + str(time_begin))
        # set3_bt2_md-unet-Final_U_YS_Net_64-15-0.0001-139-0.4000-512.pkl
        # set3_bt2_md-unet-%s-%d-%.4f-%d-%.4f-%d.pkl
        unet_path = os.path.join(self.model_path, 'set3_bt2_md-unet-%s-%d-%.4f-%d-%.4f-%d.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob, self.image_size))
        unet_path = os.path.join(self.model_path, 'not_exist.pkl')

        unet_final_path = os.path.join(self.model_path, 'set3_bt2_md-unet-Final_%s-%d-%.4f-%d-%.4f-%d.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob, self.image_size))
        # unet_path = os.path.join(self.model_path, 'set2-Multi-landmark-MD_UNet-270-0.0001-189-0.5000-512.pkl')
        # unet_final_path = os.path.join(self.model_path, 'Finallandmark-landmark-U_Net-220-0.0003-154-0.0000-128.pkl')
        # old_unet_path = os.path.join(self.model_path, 'Deform_U_Net-195-0.0001-136-0.5000-512.pkl')
        # U-Net Train
        if os.path.isfile(unet_path):
            # self.unet.load_state_dict(torch.load(unet_path), map_location=torch.device('cpu'))
            self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # init_weights(self.unet, init_type='kaiming')
            init_weights(self.unet, init_type='normal')
            print('New network initiated')
        ####加载部分预训练模型
        # model = ...
        # model_dict = model.state_dict()
        # pretrained_dict = torch.load(load_name)
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # # 3. load the new state dict
        # model.load_state_dict(model_dict)

        # Train for Encoder

        lr = self.lr
        best_score_dc = 0. # 保存dc系数最高的模型
        small_dist = 30
        avg_cost = np.zeros([self.num_epochs, 2], dtype=np.float32)
        lambda_weight = np.ones([2, self.num_epochs])
        self.unet.iter_num = self.EM_iternum

        writer = SummaryWriter('runs-AOP/AOP')
        T = 2
        model_result_path = './result/csv4000/{}/'.format(self.model_type) #为每个模型进行测试并保存
        if  os.path.exists(model_result_path) is False:
            os.makedirs(model_result_path)
            print('succeed to mkdirs: {}'.format(model_result_path))

        model_save_path = './result/model_weight_4000/{}_{}_{}'.format(self.model_type, self.lr, self.batch_size)

        for epoch in range(self.num_epochs):

            self.unet.train(True)
            epoch_loss = 0
            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            DC_1 = 0.
            DC_2 = 0.
            length = 0

            dist_1 = 0.
            dist_2 = 0.
            angle_div = 0.

            for i, (images, image2, image3, image4, GT, GT5, GT_Lmark) in enumerate(self.train_loader):
                print("epcoh[{}]:training in step:{}/{}".format(epoch, i + 1, len(self.train_loader)))
                # if i + 1 == 1:
                #     step_time1 = time.time()
                # if i + 1 == 11:
                #     step_time2 = time.time()
                #     step_time2 = step_time2 - step_time1
                #     print("10轮耗时： ", end=" ")
                #     print("%s:%s:%s" % (int(step_time2 / 3600), int(step_time2 / 60), int(step_time2 % 60)))
                #     print("1个epoch耗时预计： ", end=" ")
                #     step_time2 = step_time2 * 10
                #     print("%s:%s:%s" % (int(step_time2 / 3600), int(step_time2 / 60), int(step_time2 % 60)))
                # GT : Ground Truth

                # images = self.img_catdist_channel(images)
                images = images.to(self.device)
                image2 = image2.to(self.device)
                image3 = image3.to(self.device)
                image4 = image4.to(self.device)

                # GT_1 = F.interpolate(GT, scale_factor=0.5)
                # GT_2 = F.interpolate(GT, scale_factor=0.25)

                GT = GT.to(self.device, torch.long)
                GT_Lmark = GT_Lmark.to(self.device)
                # # GT_class = GT_class.to(self.device, torch.long)
                GT_Lmark_d2 = F.interpolate(GT_Lmark, scale_factor=0.5, mode='bilinear')
                GT_Lmark_d4 = F.interpolate(GT_Lmark, scale_factor=0.25, mode='bilinear')
                GT_Lmark_d8 = F.interpolate(GT_Lmark, scale_factor=0.125, mode='bilinear')
                GT_Lmark_d2 = GT_Lmark_d2.to(self.device)
                GT_Lmark_d4 = GT_Lmark_d4.to(self.device)
                GT_Lmark_d8 = GT_Lmark_d8.to(self.device)
                if False:
                    # GT2 = GT2.to(self.device, torch.long)
                    # GT3 = GT3.to(self.device, torch.long)
                    # GT4 = GT4.to(self.device, torch.long)
                    GT5 = GT5.to(self.device, torch.long)
                # GT_sg = self.onehot_to_mulchannel(GT)  # 转换GT编码方式
                GT_sg = self.onehot_to_mulchannel_sun(GT)
                if False:
                    # GT_sg2 = self.onehot_to_mulchannel(GT2)  # 转换GT编码方式
                    # GT_sg3 = self.onehot_to_mulchannel(GT3)  # 转换GT编码方式
                    # GT_sg4 = self.onehot_to_mulchannel(GT4)  # 转换GT编码方式
                    GT_sg5 = self.onehot_to_mulchannel(GT5)  # 转换GT编码方式
                # print(GT_sg.shape)
                # GT = GT.squeeze(1)

                # MDUNET
                # SR_lm, SR_seg, SR_lm_d2, SR_lm_d4, SR_lm_d8, logsigma = self.unet(images)
                # SR_lm = torch.sigmoid(SR_lm)
                # SR_lm_d2 = torch.sigmoid(SR_lm_d2)
                # SR_lm_d4 = torch.sigmoid(SR_lm_d4)
                # SR_lm_d8 = torch.sigmoid(SR_lm_d8)

                # loss_seg = self.criterion3(SR_seg, GT_sg)
                # loss_lm = self.criterion2(SR_lm, GT_Lmark) + 0.8 * self.criterion2(SR_lm_d2,
                #                                                                    GT_Lmark_d2) + 0.8 * self.criterion2(
                #     SR_lm_d4, GT_Lmark_d4)
                # loss = loss_lm + loss_seg
                # MDUnet end

                # u2net时候 网络输出是7个 其中第一个是融合结果 后面几个主要是为了计算loss
                # SR_seg, SR_seg5 = self.unet(images)
                # self.criterion3.set_epoch(epoch + 1)

                SR_seg = self.unet(images)
                self.criterion3.set_epoch(epoch + 1)
                # ---- loss function ---- 普通单个返回如unet
                # loss = self.criterion3(SR_seg, GT_sg)

                # u2net loss  only multi  u2net时候 网络输出是7个 其中第一个是融合结果 后面几个主要是为了计算loss
                # loss = u2net_loss(SR_seg, GT_sg)

                #saunet loss multi+shape
                GT_edge = getAllTour(GT_sg)
                loss = self.dualLoss(SR_seg, (GT_sg, GT_edge))

                #only has shape 目前sau和mynet
                # GT_edge = getAllTour(GT_sg)
                # loss = self.dualLoss_shape(SR_seg, (GT_sg, GT_edge))


                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                #Multi + shape 返回的tuple
                SR_seg_0, _, _, _, _, _, _, _ = SR_seg

                # unet返回一个
                # SR_seg_0 = SR_seg

                # only shape
                # SR_seg_0, _ = SR_seg

                # only multi
                # SR_seg_0, _,  _, _, _, _ = SR_seg

                SR_seg_0 = torch.softmax(SR_seg_0, dim=1)



                GT = GT.squeeze(1)

                # SR_seg_0 = SR_seg[0]
                # SR_seg_0 = torch.unsqueeze(SR_seg[0], dim=0) # u2net用到
                acc += get_accuracy(SR_seg_0, GT)
                sensitivity = get_sensitivity(SR_seg_0, GT)
                SE += sensitivity
                # print("get_sensitivity(SR_seg, GT): ", end="")
                # print(sensitivity)
                SP += get_specificity(SR_seg_0, GT)
                PC += get_precision(SR_seg_0, GT)
                F1 += get_F1(SR_seg_0, GT)
                JS += get_JS(SR_seg_0, GT)
                # DC += get_DC(SR_seg,GT)
                dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg_0, GT)
                print("dc:{}, dc_1:{}, dc_2:{}".format(dc_ca0, dc_ca1, dc_ca2))
                # DC += get_DC(SR_seg,GT_head)
                DC += dc_ca0
                DC_1 += dc_ca1
                DC_2 += dc_ca2
                # length += images.size(0)
                length += 1
            if length < 1:
                length = 1
            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length
            DC_1 = DC_1 / length
            DC_2 = DC_2 / length
            print('-' * 100)
            time_temp = time.time() - time_begin_second
            # time_temp = "%s:%s:%s" % (int(time_temp / 3600), int(time_temp / 60), int(time_temp % 60))
            time_temp = float(time_temp / 3600)
            print(
                'Epoch [%d/%d-%.3f], Loss: %.6f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
                    epoch + 1, self.num_epochs, time_temp, \
                    epoch_loss, \
                    acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2))
            info = (epoch+1, time_temp, epoch_loss, acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2)
            # dfhistory_train.loc[len(dfhistory_train)] = info
            dfhistory_train.loc[epoch] = info
            model_result_path_train = os.path.join(model_result_path, '{}_{}_{}_train_history.csv'.format(
                self.model_type, self.batch_size,self.lr))
            dfhistory_train.to_csv(model_result_path_train, index=False)
            # print('Epoch [%d/%d], Loss: %.6f' %(epoch+1, self.num_epochs, epoch_loss))
            # Decay learning rate
            if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))

            # ===================================== Validation ====================================#
            print('------------------------val------------------------------')
            self.unet.train(False)
            self.unet.eval()
            # self.unet.iter_num = 5
            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            DC_1 = 0.
            DC_2 = 0.
            length = 0

            # for i, (images, GT, GT_Lmark) in enumerate(self.valid_loader):
            with torch.no_grad(): # 测试时候不需要梯度回传 节省空间
                for i, (image, image2, image3, image4, GT, GT5, GT_Lmark) in enumerate(self.valid_loader):
                    # images = self.img_catdist_channel(images)
                    images = image.to(self.device)
                    # image2 = image2.to(self.device)
                    # image3 = image3.to(self.device)
                    # image4 = image4.to(self.device)
                    GT = GT.to(self.device, torch.long)


                    GT_Lmark = GT_Lmark.to(self.device)
                    # GT_class = GT_class.to(self.device,torch.long)
                    # GT = F.interpolate(GT, scale_factor=0.5, mode='bilinear').long()
                    # GT_1 = (GT == 1).squeeze(1).to(self.device)

                    GT = GT.squeeze(1)

                    # SR_lm = self.unet(images)
                    if False:
                        # SR_seg, SR_seg2, SR_seg3, SR_seg4, SR_seg5 = self.unet(images)
                        # SR_seg, SR_seg5 = self.unet(images)
                        SR_seg, SR_seg5 = self.unet(images, image2, image3, image4)
                    #MDUNET
                    # SR_lm, SR_seg, _, _, _, _ = self.unet(images)
                    #单个返回
                    SR_seg = self.unet(images)
                    # SR_seg, SR_seg5 = self.unet(images)
                    # SR_seg, g1, g2, g3= self.unet(images)
                    # SR ,_,_,_,_,_= self.unet(images)
                    # SR,d1,d2,d3 = self.unet(images)
                    # ##多分支网络
                    # SR_lm = torch.sigmoid(SR_lm)
                    # SR_seg = torch.sigmoid(SR_seg)
                    SR_seg = F.softmax(SR_seg, dim=1)
                    # print(class_out)
                    # class_out = torch.softmax(class_out,dim=1)
                    # print('after:',class_out)
                    # SR = torch.cat((SR1, SR1, SR2), dim=1)
                    # ####
                    # SR = F.softmax(SR,dim=1)		#用lovaszloss时不用加sigmoid
                    # GT = GT.squeeze(1)
                    acc += get_accuracy(SR_seg, GT)
                    SE += get_sensitivity(SR_seg, GT)
                    SP += get_specificity(SR_seg, GT)
                    PC += get_precision(SR_seg, GT)
                    F1 += get_F1(SR_seg, GT)
                    JS += get_JS(SR_seg, GT)
                    # DC += get_DC(SR_seg,GT)
                    dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg, GT)
                    DC += dc_ca0
                    # DC += get_DC(SR_seg, GT)
                    DC_1 += dc_ca1
                    DC_2 += dc_ca2
                    # length += images.size(0)
                    #mdunet
                    dist1, dist2, angle_d = get_diatance(SR_lm, GT_Lmark)
                    # dist_1 += dist1
                    # dist_2 += dist2
                    # angle_div += angle_d

                    length += 1

            # _, preds = class_out.max(1)
            # print(preds)
            # class_acc += preds.eq(GT_class).sum()/class_out.size(0)
            # print('class:',class_acc)

            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length
            DC_1 = DC_1 / length
            DC_2 = DC_2 / length


            # MDUnet
            # dist_1 = dist_1 / length
            # dist_2 = dist_2 / length
            # angle_div = angle_div / length


            # class_acc_sum = class_acc.float() / length
            # print('class_acc_sum:',class_acc_sum)
            # unet_score = DC
            # print('class_acc_sum: %.4f' % class_acc_sum)
            # print('[valid-dist] Dist1: %.4f, Dist2: %.4f, Angle_div: %.4f' % (dist_1, dist_2, angle_div))
            # writer.add_scalars('set3_bt2_md-unet-landmark/valid_dist',
            #                    {'dist_1': dist_1, 'dist_2': dist_2, 'angle_div': angle_div}, epoch)
            # writer.add_scalars('set3_bt2_md-unet-landmark/valid_DCgroup', {'DC': DC,
            #                                                                'DC_SH': DC_1,
            #                                                                'DC_Head': DC_2}, epoch)
            print(
                '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f, %.4f' % (
                    acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2, angle_div))
            info = (epoch, acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2, angle_div)
            dfhistory_valid.loc[epoch] = info
            model_result_path_valid = os.path.join(model_result_path, '{}_{}_{}_valid_history.csv'.format(
                self.model_type, self.batch_size,self.lr))

            dfhistory_valid.to_csv(model_result_path_valid, index=False)



            # temp_path = os.path.join(self.model_path, 'set3_bt2_md-unet-%s-%d-%.4f-%d-%.4f-%d.pkl' % (
            #     self.model_type, epoch, self.lr, self.num_epochs_decay, self.augmentation_prob, self.image_size))

            # yu 的思路 每个epoch模型都保存 这样太多了 磁盘会满要定时清理
            #temp_path = os.path.join(self.model_path, 'set3_bt2_md-unet-%s-%d-%.4f-%.4f-%d-%.4f-%d.pkl' % (
                #self.model_type, epoch, DC, self.lr, self.num_epochs_decay, self.augmentation_prob, self.image_size))
            # torch.save(self.unet.state_dict(), temp_path)


            if DC > best_score_dc:
                best_score_dc = DC
                # DC = round(DC, 4) #保留四位小数
                # model_save_path_pth = model_save_path + '_dc{}.pth'.format(DC) #添加dice系数
                model_save_path_pth = model_save_path + '.pth' #不添加dice系数
                torch.save(self.unet.state_dict(), model_save_path_pth)

            #查看在test上表现
            # print('--------------------------test-------------------------')
            # self.unet.train(False)
            # self.unet.eval()
            # # self.unet.iter_num = 5
            # acc = 0.  # Accuracy
            # SE = 0.  # Sensitivity (Recall)
            # SP = 0.  # Specificity
            # PC = 0.  # Precision
            # F1 = 0.  # F1 Score
            # JS = 0.  # Jaccard Similarity
            # DC = 0.  # Dice Coefficient
            # DC_1 = 0.
            # DC_2 = 0.
            # length = 0
            #
            # # for i, (images, GT, GT_Lmark) in enumerate(self.valid_loader):
            # with torch.no_grad(): # 测试时候不需要梯度回传 节省空间
            #     #for i, (image, image2, image3, image4, GT, GT5, GT_Lmark) in enumerate(self.test_loader):
            #     for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):
            #         # images = self.img_catdist_channel(images)
            #         images = images.to(self.device)
            #         # image2 = image2.to(self.device)
            #         # image3 = image3.to(self.device)
            #         # image4 = image4.to(self.device)
            #         GT = GT.to(self.device, torch.long)
            #         # GT_Lmark = GT_Lmark.to(self.device)
            #         # GT_class = GT_class.to(self.device,torch.long)
            #         # GT = F.interpolate(GT, scale_factor=0.5, mode='bilinear').long()
            #         # GT_1 = (GT == 1).squeeze(1).to(self.device)
            #
            #         GT = GT.squeeze(1)
            #
            #         # SR_lm = self.unet(images)
            #         if False:
            #             # SR_seg, SR_seg2, SR_seg3, SR_seg4, SR_seg5 = self.unet(images)
            #             # SR_seg, SR_seg5 = self.unet(images)
            #             SR_seg, SR_seg5 = self.unet(images, image2, image3, image4)
            #         # SR_seg, SR_seg5 = self.unet(images)
            #         SR_seg= self.unet(images)
            #         # SR ,_,_,_,_,_= self.unet(images)
            #         # SR,d1,d2,d3 = self.unet(images)
            #         # ##多分支网络
            #         # SR_lm = torch.sigmoid(SR_lm)
            #         # SR_seg = torch.sigmoid(SR_seg)
            #         SR_seg = F.softmax(SR_seg, dim=1)
            #         # print(class_out)
            #         # class_out = torch.softmax(class_out,dim=1)
            #         # print('after:',class_out)
            #         # SR = torch.cat((SR1, SR1, SR2), dim=1)
            #         # ####
            #         # SR = F.softmax(SR,dim=1)		#用lovaszloss时不用加sigmoid
            #         # GT = GT.squeeze(1)
            #         acc += get_accuracy_test(SR_seg, GT)
            #         SE += get_sensitivity(SR_seg, GT)
            #         SP += get_specificity(SR_seg, GT)
            #         PC += get_precision(SR_seg, GT)
            #         F1 += get_F1(SR_seg, GT)
            #         JS += get_JS(SR_seg, GT)
            #         # DC += get_DC(SR_seg,GT)
            #         dc_ca0, dc_ca1, dc_ca2 = get_DC_test(SR_seg, GT)
            #         DC += dc_ca0
            #         # DC += get_DC(SR_seg, GT)
            #         DC_1 += dc_ca1
            #         DC_2 += dc_ca2
            #         # length += images.size(0)
            #         length += 1
            #
            # acc = acc / length
            # SE = SE / length
            # SP = SP / length
            # PC = PC / length
            # F1 = F1 / length
            # JS = JS / length
            # DC = DC / length
            # DC_1 = DC_1 / length
            # DC_2 = DC_2 / length
            # print(
            #     '[Test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
            #         acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2))
            # info = (epoch, acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2)
            # dfhistory_test.loc[epoch] = info
            # model_result_path_test = os.path.join(model_result_path, '{}_{}_{}_test_history_debug.csv'.format(
            #     self.model_type, self.batch_size,self.lr))
            #
            # dfhistory_test.to_csv(model_result_path_test, index=False)
            # print("Test end at:" + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            # print("=" * 50)


            '''
            torchvision.utils.save_image(images.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(SR.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(GT.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
            '''

            # unet_score = small_dist
            # Save Best U-Net model
            # if unet_score > best_unet_score:
            # 	best_unet_score = unet_score
            # 	best_epoch = epoch
            # 	best_unet = self.unet.state_dict()
            # 	print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
            # 	torch.save(best_unet,unet_path)
            ####landmark部分
            # if unet_score < small_dist:
            #     small_dist = unet_score
            #     best_epoch = epoch
            #     best_unet = self.unet.state_dict()
            #     print('Best %s model score : %.4f' % (self.model_type, small_dist))
            #     torch.save(best_unet, unet_path)
        # torch.save(self.unet.state_dict(), unet_final_path)
        print("Train end at:" + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        print("=" * 50)

        #旧
        # writer.close()
        # ===================================== Test ====================================#
        # del self.unet
        # # del best_unet
        # self.build_model()
        # self.unet.load_state_dict(torch.load(unet_final_path))
        #
        # self.unet.train(False)
        # self.unet.eval()
        #
        # acc = 0.  # Accuracy
        # SE = 0.  # Sensitivity (Recall)
        # SP = 0.  # Specificity
        # PC = 0.  # Precision
        # F1 = 0.  # F1 Score
        # JS = 0.  # Jaccard Similarity
        # DC = 0.  # Dice Coefficient
        # DC_1 = 0.
        # DC_2 = 0.
        # length = 0

    def sun_get_test_result(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        time_begin_second = time.time()
        time_begin = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_begin_second))


        dfhistory_test = pd.DataFrame(
            columns=["epoch", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "DC1", "DC2", "angle_div"])
        # dfhistory_test = pd.DataFrame(
        #     columns=["epoch", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "DC1", "DC2"])
        # dfhistory_train = pd.read_csv('./result/csv/train_history.csv')
        # dfhistory_valid = pd.read_csv('./result/csv/valid_history.csv')
        print('=' * 50)
        print("Train start at:" + str(time_begin))
        # set3_bt2_md-unet-Final_U_YS_Net_64-15-0.0001-139-0.4000-512.pkl
        # set3_bt2_md-unet-%s-%d-%.4f-%d-%.4f-%d.pkl


        unet_path = './result/model_weight_4000/{}_0.0001_1.pth'.format(self.model_type)
        if os.path.isfile(unet_path):
            # self.unet.load_state_dict(torch.load(unet_path), map_location=torch.device('cpu'))
            self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cuda')))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # init_weights(self.unet, init_type='kaiming')
            init_weights(self.unet, init_type='normal')
            print('New network initiated')

        model_result_path = './result/csv4000/{}/'.format(self.model_type) #为每个模型进行测试并保存
        if  os.path.exists(model_result_path) is False:
            os.makedirs(model_result_path)
            print('succeed to mkdirs: {}'.format(model_result_path))


        print('------------------------test------------------------------')
        self.unet.train(False)
        self.unet.eval()
        # self.unet.iter_num = 5
        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        DC_1 = 0.
        DC_2 = 0.
        length = 0
        angle_div = 0

        # for i, (images, GT, GT_Lmark) in enumerate(self.valid_loader):
        with torch.no_grad(): # 测试时候不需要梯度回传 节省空间
            # for i, (image, image2, image3, image4, GT, GT5, GT_Lmark) in enumerate(self.test_loader):
            for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):
                # images = self.img_catdist_channel(images)
                images = images.to(self.device)
                # image2 = image2.to(self.device)
                # image3 = image3.to(self.device)
                # image4 = image4.to(self.device)
                GT = GT.to(self.device, torch.long)


                GT_Lmark = GT_Lmark.to(self.device)
                # GT_class = GT_class.to(self.device,torch.long)
                # GT = F.interpolate(GT, scale_factor=0.5, mode='bilinear').long()
                # GT_1 = (GT == 1).squeeze(1).to(self.device)

                GT = GT.squeeze(1)

                # SR_lm = self.unet(images)
                if False:
                    # SR_seg, SR_seg2, SR_seg3, SR_seg4, SR_seg5 = self.unet(images)
                    # SR_seg, SR_seg5 = self.unet(images)
                    SR_seg, SR_seg5 = self.unet(images, image2, image3, image4)
                #MDUNET
                # SR_lm, SR_seg, _, _, _, _ = self.unet(images)
                #单个返回
                SR_seg = self.unet(images)
                # SR_seg, SR_seg5 = self.unet(images)
                # SR_seg, g1, g2, g3= self.unet(images)
                # SR ,_,_,_,_,_= self.unet(images)
                # SR,d1,d2,d3 = self.unet(images)
                # ##多分支网络
                # SR_lm = torch.sigmoid(SR_lm)
                # SR_seg = torch.sigmoid(SR_seg)
                SR_seg = F.softmax(SR_seg, dim=1)
                # print(class_out)
                # class_out = torch.softmax(class_out,dim=1)
                # print('after:',class_out)
                # SR = torch.cat((SR1, SR1, SR2), dim=1)
                # ####
                # SR = F.softmax(SR,dim=1)		#用lovaszloss时不用加sigmoid
                # GT = GT.squeeze(1)
                acc += get_accuracy(SR_seg, GT)
                SE += get_sensitivity(SR_seg, GT)
                SP += get_specificity(SR_seg, GT)
                PC += get_precision(SR_seg, GT)
                F1 += get_F1(SR_seg, GT)
                JS += get_JS(SR_seg, GT)
                # DC += get_DC(SR_seg,GT)
                dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg, GT)
                DC += dc_ca0
                # DC += get_DC(SR_seg, GT)
                DC_1 += dc_ca1
                DC_2 += dc_ca2
                # length += images.size(0)
                #mdunet
                # dist1, dist2, angle_d = get_diatance(SR_lm, GT_Lmark)
                # dist_1 += dist1
                # dist_2 += dist2
                # angle_div += angle_d

                length += 1

        # _, preds = class_out.max(1)
        # print(preds)
        # class_acc += preds.eq(GT_class).sum()/class_out.size(0)
        # print('class:',class_acc)

        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        DC_1 = DC_1 / length
        DC_2 = DC_2 / length


        # MDUnet
        # dist_1 = dist_1 / length
        # dist_2 = dist_2 / length
        # angle_div = angle_div / length


        # class_acc_sum = class_acc.float() / length
        # print('class_acc_sum:',class_acc_sum)
        # unet_score = DC
        # print('class_acc_sum: %.4f' % class_acc_sum)
        # print('[valid-dist] Dist1: %.4f, Dist2: %.4f, Angle_div: %.4f' % (dist_1, dist_2, angle_div))
        # writer.add_scalars('set3_bt2_md-unet-landmark/valid_dist',
        #                    {'dist_1': dist_1, 'dist_2': dist_2, 'angle_div': angle_div}, epoch)
        # writer.add_scalars('set3_bt2_md-unet-landmark/valid_DCgroup', {'DC': DC,
        #                                                                'DC_SH': DC_1,
        #                                                                'DC_Head': DC_2}, epoch)
        print(
            '[test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f, %.4f' % (
                acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2, angle_div))
        epoch = 1
        info = (epoch, acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2, angle_div)
        dfhistory_test.loc[epoch] = info
        model_result_path_test = os.path.join(model_result_path, '{}_{}_{}_test_history.csv'.format(
            self.model_type, self.batch_size,self.lr))

        dfhistory_test.to_csv(model_result_path_test, index=False)
        # torch.save(self.unet.state_dict(), unet_final_path)
        print("test end at:" + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        print("=" * 50)

        #旧
        # writer.close()
        # ===================================== Test ====================================#
        # del self.unet
        # # del best_unet
        # self.build_model()
        # self.unet.load_state_dict(torch.load(unet_final_path))
        #
        # self.unet.train(False)
        # self.unet.eval()
        #
        # acc = 0.  # Accuracy
        # SE = 0.  # Sensitivity (Recall)
        # SP = 0.  # Specificity
        # PC = 0.  # Precision
        # F1 = 0.  # F1 Score
        # JS = 0.  # Jaccard Similarity
        # DC = 0.  # Dice Coefficient
        # DC_1 = 0.
        # DC_2 = 0.
        # length = 0

    def yu_get_test_result(self, fold = 'd1'):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        time_begin_second = time.time()
        time_begin = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_begin_second))


        dfhistory_test = pd.DataFrame(
            columns=["epoch", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "DC1", "DC2", "angle_div", "Asd"])
        dfhistory_test_single = pd.DataFrame(
            columns=["pic_no", "acc", "DC", "DC1", "DC2", "Asd"])
        # dfhistory_test = pd.DataFrame(
        #     columns=["epoch", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "DC1", "DC2"])
        # dfhistory_train = pd.read_csv('./result/csv/train_history.csv')
        # dfhistory_valid = pd.read_csv('./result/csv/valid_history.csv')
        print('=' * 50)
        print("Train start at:" + str(time_begin))
        # set3_bt2_md-unet-Final_U_YS_Net_64-15-0.0001-139-0.4000-512.pkl
        # set3_bt2_md-unet-%s-%d-%.4f-%d-%.4f-%d.pkl


        unet_path = './result/model_weight_4000/{}_{}.pth'.format(self.model_type, fold)
        if os.path.isfile(unet_path):
            # self.unet.load_state_dict(torch.load(unet_path), map_location=torch.device('cpu'))
            self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cuda')))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # init_weights(self.unet, init_type='kaiming')
            # init_weights(self.unet, init_type='normal')
            # print('New network initiated')
            print('load model error: path not exist')
            return

        model_result_path = './result/csv4000/{}/{}/'.format(self.model_type, fold) #为每个模型进行测试并保存
        if  os.path.exists(model_result_path) is False:
            os.makedirs(model_result_path)
            print('succeed to mkdirs: {}'.format(model_result_path))


        print('------------------------test------------------------------')
        self.unet.train(False)
        self.unet.eval()
        # self.unet.iter_num = 5
        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        DC_1 = 0.
        DC_2 = 0.
        length = 0
        angle_div = 0
        Asd = 0
        count = 0

        # for i, (images, GT, GT_Lmark) in enumerate(self.valid_loader):
        with torch.no_grad(): # 测试时候不需要梯度回传 节省空间
            # for i, (image, image2, image3, image4, GT, GT5, GT_Lmark) in enumerate(self.test_loader):
            for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):
                # images = self.img_catdist_channel(images)
                pic_no = 'ATD-marked_result-{}'.format(filename)
                images = images.to(self.device)
                # image2 = image2.to(self.device)
                # image3 = image3.to(self.device)
                # image4 = image4.to(self.device)
                GT = GT.to(self.device, torch.long)


                GT_Lmark = GT_Lmark.to(self.device)
                # GT_class = GT_class.to(self.device,torch.long)
                # GT = F.interpolate(GT, scale_factor=0.5, mode='bilinear').long()
                # GT_1 = (GT == 1).squeeze(1).to(self.device)

                GT = GT.squeeze(1)

                # SR_lm = self.unet(images)
                if False:
                    # SR_seg, SR_seg2, SR_seg3, SR_seg4, SR_seg5 = self.unet(images)
                    # SR_seg, SR_seg5 = self.unet(images)
                    SR_seg, SR_seg5 = self.unet(images, image2, image3, image4)
                #MDUNET
                # SR_lm, SR_seg, _, _, _, _ = self.unet(images)
                #单个返回
                SR_seg = self.unet(images)
                # SR_seg, SR_seg5 = self.unet(images)
                # SR_seg, g1, g2, g3= self.unet(images)
                # SR ,_,_,_,_,_= self.unet(images)
                # SR,d1,d2,d3 = self.unet(images)
                # ##多分支网络
                # SR_lm = torch.sigmoid(SR_lm)
                # SR_seg = torch.sigmoid(SR_seg)
                SR_seg = F.softmax(SR_seg, dim=1)
                # print(class_out)
                # class_out = torch.softmax(class_out,dim=1)
                # print('after:',class_out)
                # SR = torch.cat((SR1, SR1, SR2), dim=1)
                # ####
                # SR = F.softmax(SR,dim=1)		#用lovaszloss时不用加sigmoid
                # GT = GT.squeeze(1)
                # acc += get_accuracy(SR_seg, GT)

                temp_acc = get_accuracy(SR_seg, GT)
                acc += temp_acc

                SE += get_sensitivity(SR_seg, GT)
                SP += get_specificity(SR_seg, GT)
                PC += get_precision(SR_seg, GT)
                F1 += get_F1(SR_seg, GT)
                JS += get_JS(SR_seg, GT)
                # DC += get_DC(SR_seg,GT)
                dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg, GT)
                DC += dc_ca0
                # DC += get_DC(SR_seg, GT)
                DC_1 += dc_ca1
                DC_2 += dc_ca2

                all_asdd = 0
                pred = torch.argmax(SR_seg, dim=1)[0]
                pred [pred != 0] = 1

                pred_cpu = pred.cpu()

                pred = np.asarray(pred_cpu, dtype=np.bool)
                label = GT[0]
                label[label != 0] =1
                label_cpu = label.cpu()
                label = np.asarray(label_cpu, dtype=np.bool)
                surface_distance_all = surface_distance.compute_surface_distances(pred, label, spacing_mm=(1.0,1.0))
                all_asdd = surface_distance.compute_average_surface_distance(surface_distance_all)  # ASD
                all_asdd = math.sqrt(all_asdd[0] * all_asdd[0] + all_asdd[1] * all_asdd[1])
                all_hd_100 = surface_distance.compute_robust_hausdorff(surface_distance_all, 100)  # HD_100
                Asd += all_asdd
                # length += images.size(0)
                #mdunet
                # dist1, dist2, angle_d = get_diatance(SR_lm, GT_Lmark)
                # dist_1 += dist1
                # dist_2 += dist2
                # angle_div += angle_d

                length += 1
                temp_info = (pic_no, temp_acc, dc_ca0, dc_ca1, dc_ca2, all_asdd)
                dfhistory_test_single.loc[count] = temp_info
                count = count + 1
                print(count)
                dfhistory_test_single.to_csv(model_result_path + 'dice_asd_single.csv', index=False)

        # _, preds = class_out.max(1)
        # print(preds)
        # class_acc += preds.eq(GT_class).sum()/class_out.size(0)
        # print('class:',class_acc)

        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        DC_1 = DC_1 / length
        DC_2 = DC_2 / length
        Asd = Asd / length


        # MDUnet
        # dist_1 = dist_1 / length
        # dist_2 = dist_2 / length
        # angle_div = angle_div / length


        # class_acc_sum = class_acc.float() / length
        # print('class_acc_sum:',class_acc_sum)
        # unet_score = DC
        # print('class_acc_sum: %.4f' % class_acc_sum)
        # print('[valid-dist] Dist1: %.4f, Dist2: %.4f, Angle_div: %.4f' % (dist_1, dist_2, angle_div))
        # writer.add_scalars('set3_bt2_md-unet-landmark/valid_dist',
        #                    {'dist_1': dist_1, 'dist_2': dist_2, 'angle_div': angle_div}, epoch)
        # writer.add_scalars('set3_bt2_md-unet-landmark/valid_DCgroup', {'DC': DC,
        #                                                                'DC_SH': DC_1,
        #                                                                'DC_Head': DC_2}, epoch)
        print(
            '[test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f, %.4f' % (
                acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2, angle_div))
        epoch = 1
        info = (epoch, acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2, angle_div, Asd)
        dfhistory_test.loc[epoch] = info
        model_result_path_test = os.path.join(model_result_path, '{}_{}_{}_test_history.csv'.format(
            self.model_type, self.batch_size,self.lr))

        dfhistory_test.to_csv(model_result_path_test, index=False)
        # torch.save(self.unet.state_dict(), unet_final_path)
        print("test end at:" + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        print("=" * 50)

        #旧
        # writer.close()
        # ===================================== Test ====================================#
        # del self.unet
        # # del best_unet
        # self.build_model()
        # self.unet.load_state_dict(torch.load(unet_final_path))
        #
        # self.unet.train(False)
        # self.unet.eval()
        #
        # acc = 0.  # Accuracy
        # SE = 0.  # Sensitivity (Recall)
        # SP = 0.  # Specificity
        # PC = 0.  # Precision
        # F1 = 0.  # F1 Score
        # JS = 0.  # Jaccard Similarity
        # DC = 0.  # Dice Coefficient
        # DC_1 = 0.
        # DC_2 = 0.
        # length = 0

    def yu_get_generate_output_pic(self, mode = 'valid',dataset_no=99, fold = 'd1'):
        if dataset_no == 99 and mode == 'valid':
            return
        # set3_bt2_md-unet-Final_U_YS_Net_16-21-0.0001-139-0.3000-512.pkl
        # set3_bt2_md-unet-Final_U_YS_Net_16-14-0.0001-139-0.3000-512.pkl


        #加载模型直接测试
        # set3_bt2_md-unet-MBU_YS_Net_16-4-0.0001-139-0.3000-512.pkl
        # unet_path = os.path.join(self.model_path, 'final_models/dataset1/set3_bt2_md-unet-DBU_YS_Net_16-32-0.0001-139-0.3000-512.pkl')
        #### /models/templmodels/set3 bt2 md-unet-fcnlresnet50-49-0.0001-139-0.30vm0-512.pkl
        # unet_path = './models/temp_models/set3_bt2_md-unet-fcn_resnet50-49-0.0001-139-0.3000-512.pkl'
        unet_path = 'hh./models/temp_models/set3_bt2_md-unet-U_Net-299-0.9078-0.0000-139-0.3000-51.pklhhh' #3chu
        #unet_path = './result/model_weight_4000/U_Net_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/My_Unet_has_multi_has_shape_no_attention_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/AttU_Net_0.0001_1_dc0.9384.pth'

        # unet_path = './result/model_weight_4000/Multi_U_Net_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/My_Unet_no_multi_has_shape_no_attention_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/My_Unet_has_multi_has_shape_no_attention_d3_0.0001_1.pth111'
        # unet_path = './result/model_weight_4000/AG_DBU_YS_Net_16_0.0001_1.pth'
        unet_path = './result/model_weight_4000/{}_{}.pth'.format(self.model_type, fold)
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')
            return

        # test读取测试集313张图片 否则读取5折中的验证集合的图片
        # if mode == 'test':
        #     test_path = './dataset_test/test'
        # else:
        #     test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # temp_test_path = './dataset_{}/valid/valid'.format(dataset_no)
        # temp_test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # temp_test_path = './dataset_test/test'.format(dataset_no)
        temp_image_size = 512
        temp_batch_size = 1
        temp_num_workers = 1
        # self.test_loader = get_loader(image_path=test_path,
        #                               image_size=temp_image_size,
        #                               batch_size=temp_batch_size,
        #                               num_workers=temp_num_workers,
        #                               mode='test',
        #                               augmentation_prob=0.)
        with torch.no_grad(): # 测试时候不需要梯度回传 节省空间
            self.unet.train(False)
            self.unet.eval()
            # 读取深度文件
            # depth_Path = 'D:/py_seg/Landmark-Net/depth.csv'
            # landmark_depth = np.loadtxt(depth_Path, delimiter=',')
            pixel_num = self.image_size * 0.715
            a_5 = 0
            a_10 = 0
            length = 0
            # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
            list_aop = []
            list_l = []
            list_r = []
            list_aop_root_mse = []
            out_aop = []
            true_aop = []
            for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):
                # degth_cm = landmark_depth[num - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                image = images
                # images = self.img_catdist_channel(images)
                images = images.to(self.device)
                GT = GT.to(self.device, torch.long)
                # SR = F.sigmoid(self.unet(images))
                # t1 = time.time()
                # SR_lm = self.unet(images)
                # SR = F.sigmoid(SR)
                # SR2 = F.sigmoid(SR_r)
                # SR = torch.cat((SR_l, SR_l, SR_r), dim=1)
                # GT_sg = self.onehot_to_mulchannel(GT)
                GT = GT.squeeze(1)
                # print('max:', GT.max())

                GT_0 = GT == 0  # 0代表其余部分
                GT_1 = GT == 76  # 1代表耻骨联合
                GT_2 = GT == 150  # 2代表胎头

                # GT2 = GT2.squeeze(1)
                # GT2_0 = GT2 == 0  # 0代表其余部分
                # GT2_1 = GT2 == 1  # 1代表耻骨联合
                # GT2_2 = GT2 == 2  # 2代表胎头
                #
                # GT3 = GT3.squeeze(1)
                # GT3_0 = GT3 == 0  # 0代表其余部分
                # GT3_1 = GT3 == 1  # 1代表耻骨联合
                # GT3_2 = GT3 == 2  # 2代表胎头
                #
                # GT4 = GT4.squeeze(1)
                # GT4_0 = GT4 == 0  # 0代表其余部分
                # GT4_1 = GT4 == 1  # 1代表耻骨联合
                # GT4_2 = GT4 == 2  # 2代表胎头
                #
                # GT5 = GT5.squeeze(1)
                # GT5_0 = GT5 == 0  # 0代表其余部分
                # GT5_1 = GT5 == 1  # 1代表耻骨联合
                # GT5_2 = GT5 == 2  # 2代表胎头

                GT = torch.cat((GT_0.unsqueeze(1), GT_1.unsqueeze(1), GT_2.unsqueeze(1)), 1).int()
                # print(GT.max())
                print(cor_num)
                # yu的图片要用012 新数据用sun
                # GT = self.onehot_to_mulchannel_sun(GT)
                # GT = self.onehot_to_mulchannel(GT)


                # GT2 = torch.cat((GT2_0.unsqueeze(1), GT2_1.unsqueeze(1), GT2_2.unsqueeze(1)), 1).int()
                # GT3 = torch.cat((GT3_0.unsqueeze(1), GT3_1.unsqueeze(1), GT3_2.unsqueeze(1)), 1).int()
                # GT4 = torch.cat((GT4_0.unsqueeze(1), GT4_1.unsqueeze(1), GT4_2.unsqueeze(1)), 1).int()
                # GT5 = torch.cat((GT5_0.unsqueeze(1), GT5_1.unsqueeze(1), GT5_2.unsqueeze(1)), 1).int()

                # SR_lm, SR_seg, _, _, _, _ = self.unet(images)
                # SR_seg = self.unet(images)
                # SR_seg, SR_seg2, SR_seg3, SR_seg4, SR_seg5 = self.unet(images)
                if False:
                    SR_seg, SR_seg5 = self.unet(images)
                # SR_seg, SR_seg5 = self.unet(images)
                # SR_seg, g1, g2, g3 = self.unet(images)
                SR_seg = self.unet(images)
                # SR_seg=SR_lm
                # SR_lm = torch.sigmoid(SR_lm)
                SR_seg = torch.softmax(SR_seg, 1) > 0.5
                # SR_seg2 = torch.softmax(SR_seg2, 1) > 0.5
                # SR_seg3 = torch.softmax(SR_seg3, 1) > 0.5
                # SR_seg4 = torch.softmax(SR_seg4, 1) > 0.5
                if False:
                    SR_seg5 = torch.softmax(SR_seg5, 1) > 0.5

                SR_seg = SR_seg.int()



                Src = GT.mul(255)
                Dst = SR_seg.mul(255)
                # Dst2 = SR_seg2.mul(255)
                # Dst3 = SR_seg3.mul(255)
                # Dst4 = SR_seg4.mul(255)

                # contours, _ = cv.findContours(SR[:, :, 1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

                # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no))
                # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no))
                # temp_path = os.path.join(self.model_path, 'final_results/{}/unet_image/'.format(self.model_type))
                temp_path = os.path.join('aop_results/{}/{}/unet_image/'.format(self.model_type, fold)) #为每个模型进行测试并保存
                if  os.path.exists(temp_path) is False:
                    os.makedirs(temp_path)
                    print('succeed to mkdirs: {}'.format(temp_path))

                for index in range(0, len(filename)):
                    # for index in range(0, len(cor_num)):
                    # filename = filename[0]
                    for i in range(0, 3): # src是GT
                        # np.expand_dims(0)
                        image_item = Src[index][i].cpu().detach().numpy().astype(np.uint8)
                        image_item = Image.fromarray(image_item, mode='L')
                        # image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i)))
                        image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i)))
                        print('-------------------std-to-real-----------------------')
                        # image_item.save('./result/pictures_DBU_1_1_1_single_0.2_upsamle_defnorm_conv/ATD-{}-src-{}.png'.format(cor_num[index], i))
                    for i in range(0, 3): # dst是测试图片
                        image_item = Dst[index][i].cpu().detach().numpy().astype(np.uint8)
                        image_item = Image.fromarray(image_item, mode='L')
                        # image_item.save(os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(cor_num[index], i)))
                        image_item.save(os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i)))


                print("generate batch {}".format(len(cor_num)))
                # print(filename)


            print("generate end ......")

    def yu_get_output_aop_sun_normal(self, mode = 'valid', dataset_no=99, fold = 'd1'):
        if dataset_no == 99 and mode =='valid':
            return

        # temp_test_path = './dataset_{}/valid/valid'.format(dataset_no)
        # temp_test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # if mode == 'test':
        #     test_path = './dataset_test/test'
        # else:
        #     test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # temp_image_size = 512
        # temp_batch_size = 1
        # temp_num_workers = 1
        # self.test_loader = get_loader(image_path=test_path,
        #                               image_size=temp_image_size,
        #                               batch_size=temp_batch_size,
        #                               num_workers=temp_num_workers,
        #                               mode='test',
        #                               augmentation_prob=0.)
        # self.unet.train(False)
        # self.unet.eval()
        # 读取深度文件
        depth_Path = './csv/depth.csv'
        aopnewGT_path = './csv/aopnewGT.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        aopnewGT = np.loadtxt(aopnewGT_path, delimiter=',')
        pixel_num = self.image_size * 0.715 # why
        self.image_size_h = 384
        a_5 = 0
        a_10 = 0
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        list_aop = []
        list_l = []
        list_r = []
        list_aop_root_mse = []
        out_aop = []
        true_aop = []
        temp_path = os.path.join('aop_results/{}/{}/unet_image/'.format(self.model_type, fold)) #为每个模型进行测试并保存
        aop_result_path = temp_path.split('unet_image')[0]
        if  os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
            print('succeed to mkdirs: {}'.format(temp_path))
        # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no)) # 加了csv
        # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no)) # 加了csv
        dfhistory_valid_aop = pd.DataFrame(
            columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_distance",
                     "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff"])
        count = 0
        W = 1295
        H = 1026
        for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):

            for index in range(0, len(filename)):
                left_point = right_point = tangent_point = []
                standard_left_point = standard_right_point = standard_tangent_point = []
                # degth_cm = landmark_depth[cor_num[index] - 1]
                # aopnewGT_item = aopnewGT[cor_num[index] - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                # standard_left_point = (aopnewGT_item[0] * (self.image_size / W), aopnewGT_item[1] * (self.image_size_h / H)) #pixel_num
                # standard_right_point = (aopnewGT_item[2] * (self.image_size / W), aopnewGT_item[3] * (self.image_size_h / H)) #pixel_num
                # standard_tangent_point = (aopnewGT_item[4] * (self.image_size / W), aopnewGT_item[5] * (self.image_size_h / H)) #pixel_num

                temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], 0))
                # temp_path_item = os.path.join(temp_path, '623-{}-GT-{}.png'.format(filename[0], 0))
                img_background = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)
                img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGR)

                # 获得GT图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i))
                    # print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        break
                    standard_ellipse_instance = cv2.fitEllipse(max_contour[0])
                    # cv2.ellipse(img_background, standard_ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        standard_left_point, standard_right_point = get_endpoint(standard_ellipse_instance)
                    else:
                        standard_tangent_point = get_tangent_point(standard_right_point, standard_ellipse_instance)

                # 获得预测图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) < 5: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        flag = 1
                        break
                    ellipse_instance = cv2.fitEllipse(max_contour[0])
                    cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        left_point, right_point = get_endpoint(ellipse_instance)
                    else:
                        tangent_point = get_tangent_point(right_point, ellipse_instance)

                if flag == 1:
                    flag == 0
                    continue

                # 计算距离并保存
                # distance = pixel_mm * math.sqrt(math.pow(right_point[0] - left_point[0], 2) + math.pow(right_point[1] - left_point[1], 2))
                cv2.line(img_background, (round(left_point[0]), round(left_point[1])),
                         (round(right_point[0]), round(right_point[1])), (255, 255, 0), 1)
                # cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 2)
                cv2.line(img_background, (round(right_point[0]), round(right_point[1])),
                         (round(tangent_point[0]), round(tangent_point[1])), (0, 0, 255), 1)
                temp_path_item = os.path.join(temp_path, 'ATD-{}-real_marked_result-{}.png'.format(filename[index], 0))
                cv2.imwrite(temp_path_item, img_background)

                pic_no = 'ATD-marked_result-{}'.format(filename[index])
                left_point_pixel = math.sqrt(math.pow(left_point[0] - standard_left_point[0], 2) + math.pow(
                    left_point[1] - standard_left_point[1], 2))
                # left_point_distance = pixel_mm * left_point_pixel
                left_point_distance = left_point_pixel
                right_point_pixel = math.sqrt(math.pow(right_point[0] - standard_right_point[0], 2) + math.pow(
                    right_point[1] - standard_right_point[1], 2))
                # right_point_distance = pixel_mm * right_point_pixel
                right_point_distance =  right_point_pixel
                tangent_point_pixel = math.sqrt(math.pow(tangent_point[0] - standard_tangent_point[0], 2) + math.pow(
                    tangent_point[1] - standard_tangent_point[1], 2))
                # tangent_point_distance = pixel_mm * tangent_point_pixel
                tangent_point_distance =  tangent_point_pixel

                # 计算预测的aop角度 角度计算 a · b = |a| * |b| * cos(o)
                vector_standard_axis = (
                    standard_left_point[0] - standard_right_point[0], standard_left_point[1] - standard_right_point[1])
                vector_output_axis = (left_point[0] - right_point[0], left_point[1] - right_point[1])
                top = vector_output_axis[0] * vector_standard_axis[0] + vector_output_axis[1] * vector_standard_axis[1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                axis_angle_diff = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                vector_standard_tangent = (standard_tangent_point[0] - standard_right_point[0],
                                           standard_tangent_point[1] - standard_right_point[1])
                vector_output_tangent = (tangent_point[0] - right_point[0], tangent_point[1] - right_point[1])

                top = vector_output_axis[0] * vector_output_tangent[0] + vector_output_axis[1] * vector_output_tangent[
                    1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_output_tangent[0], 2) + math.pow(vector_output_tangent[1], 2))
                output_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                # 计算标准aop角度
                top = vector_standard_axis[0] * vector_standard_tangent[0] + vector_standard_axis[1] * \
                      vector_standard_tangent[1]
                bottom1 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_tangent[0], 2) + math.pow(vector_standard_tangent[1], 2))
                standard_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)
                # print("*********")
                print(standard_aop)
                aop_diff = math.fabs(standard_aop - output_aop)
                info = (pic_no, left_point_pixel, left_point_distance, right_point_pixel, right_point_distance,
                        tangent_point_pixel, tangent_point_distance, axis_angle_diff, aop_diff)
                # dfhistory_train.loc[len(dfhistory_train)] = info
                dfhistory_valid_aop.loc[count] = info
                count = count + 1
                dfhistory_valid_aop.to_csv(aop_result_path + 'aop_statistics.csv', index=False)
            print("generate batch {}".format(len(cor_num)))
        print("generate end ......")

    def yu_get_result_test_sun(self, fold = 'd1'):

        result_history = pd.DataFrame(
            columns=["dataset_no", "left_pixel_mean", "left_pixel_median", "left_pixel_var", "left_distance_mean", "left_distance_median", "left_distance_var",
                     "right_pixel_mean", "right_pixel_median", "right_pixel_var", "right_distance_mean", "right_distance_median", "right_distance_var",
                     "axis_diff_mean", "axis_diff_median", "axis_diff_var", "aop_diff_mean", "aop_diff_median", "aop_diff_var"]
        )
        count = 0
        for dataset_no in range(1):# final_results/new_0.715_to_0.75/AG_DBU_old/dataset{}/
            # csv_path = os.path.join(self.model_path, 'final_results/dataset{}/aop_history.csv'.format(dataset_no))
            # csv_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/aop_history.csv')

            # aop_statistics_result_path = 'aop_results/{}/aop_statistics_both.csv'.format(self.model_type) #为每个模型进行测试并保存
            aop_statistics_result_path = 'aop_results/{}/{}/aop_statistics.csv'.format(self.model_type, fold) #为每个模型进行测试并保存
            item_history = pd.read_csv(aop_statistics_result_path)
            list_pixel = item_history['left_point_pixel']
            left_pixel_mean, left_pixel_median, left_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['left_point_distance']
            left_distance_mean, left_distance_median, left_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_pixel']
            right_pixel_mean, right_pixel_median, right_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_distance']
            right_distance_mean, right_distance_median, right_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['axis_angle_diff']
            axis_diff_mean, axis_diff_median, axis_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['aop_diff']
            aop_diff_mean, aop_diff_median, aop_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))

            info = (dataset_no, left_pixel_mean, left_pixel_median, left_pixel_var, left_distance_mean, left_distance_median, left_distance_var,
                    right_pixel_mean, right_pixel_median, right_pixel_var, right_distance_mean, right_distance_median, right_distance_var,
                    axis_diff_mean, axis_diff_median, axis_diff_var, aop_diff_mean, aop_diff_median, aop_diff_var)
            result_history.loc[count] = info
            count = count + 1

        aop_analysis_path = aop_statistics_result_path.split('aop_statistics')[0]
        # result_history.to_csv(aop_analysis_path + 'aop_analysis_both.csv', index=False)
        result_history.to_csv(aop_analysis_path + 'aop_analysis.csv', index=False)
        print("genenrate result_history success ......")



    def generate_output_pic(self, mode = 'valid',dataset_no=99):
        if dataset_no == 99 and mode == 'valid':
            return
        # set3_bt2_md-unet-Final_U_YS_Net_16-21-0.0001-139-0.3000-512.pkl
        # set3_bt2_md-unet-Final_U_YS_Net_16-14-0.0001-139-0.3000-512.pkl


        #加载模型直接测试
        # set3_bt2_md-unet-MBU_YS_Net_16-4-0.0001-139-0.3000-512.pkl
        # unet_path = os.path.join(self.model_path, 'final_models/dataset1/set3_bt2_md-unet-DBU_YS_Net_16-32-0.0001-139-0.3000-512.pkl')
        #### /models/templmodels/set3 bt2 md-unet-fcnlresnet50-49-0.0001-139-0.30vm0-512.pkl
        # unet_path = './models/temp_models/set3_bt2_md-unet-fcn_resnet50-49-0.0001-139-0.3000-512.pkl'
        unet_path = 'hh./models/temp_models/set3_bt2_md-unet-U_Net-299-0.9078-0.0000-139-0.3000-51.pklhhh' #3chu
        #unet_path = './result/model_weight_4000/U_Net_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/My_Unet_has_multi_has_shape_no_attention_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/AttU_Net_0.0001_1_dc0.9384.pth'

        # unet_path = './result/model_weight_4000/Multi_U_Net_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/My_Unet_no_multi_has_shape_no_attention_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/My_Unet_has_multi_has_shape_no_attention_d3_0.0001_1.pth111'
        # unet_path = './result/model_weight_4000/AG_DBU_YS_Net_16_0.0001_1.pth'
        unet_path = './result/model_weight_4000/{}_0.0001_1.pth'.format(self.model_type)
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        # test读取测试集313张图片 否则读取5折中的验证集合的图片
        # if mode == 'test':
        #     test_path = './dataset_test/test'
        # else:
        #     test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # temp_test_path = './dataset_{}/valid/valid'.format(dataset_no)
        # temp_test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # temp_test_path = './dataset_test/test'.format(dataset_no)
        temp_image_size = 512
        temp_batch_size = 1
        temp_num_workers = 1
        # self.test_loader = get_loader(image_path=test_path,
        #                               image_size=temp_image_size,
        #                               batch_size=temp_batch_size,
        #                               num_workers=temp_num_workers,
        #                               mode='test',
        #                               augmentation_prob=0.)
        with torch.no_grad(): # 测试时候不需要梯度回传 节省空间
            self.unet.train(False)
            self.unet.eval()
            # 读取深度文件
            # depth_Path = 'D:/py_seg/Landmark-Net/depth.csv'
            # landmark_depth = np.loadtxt(depth_Path, delimiter=',')
            pixel_num = self.image_size * 0.715
            a_5 = 0
            a_10 = 0
            length = 0
            # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
            list_aop = []
            list_l = []
            list_r = []
            list_aop_root_mse = []
            out_aop = []
            true_aop = []
            for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):
                # degth_cm = landmark_depth[num - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                image = images
                # images = self.img_catdist_channel(images)
                images = images.to(self.device)
                GT = GT.to(self.device, torch.long)
                # SR = F.sigmoid(self.unet(images))
                # t1 = time.time()
                # SR_lm = self.unet(images)
                # SR = F.sigmoid(SR)
                # SR2 = F.sigmoid(SR_r)
                # SR = torch.cat((SR_l, SR_l, SR_r), dim=1)
                # GT_sg = self.onehot_to_mulchannel(GT)
                GT = GT.squeeze(1)
                # print('max:', GT.max())

                GT_0 = GT == 0  # 0代表其余部分
                GT_1 = GT == 76  # 1代表耻骨联合
                GT_2 = GT == 150  # 2代表胎头

                # GT2 = GT2.squeeze(1)
                # GT2_0 = GT2 == 0  # 0代表其余部分
                # GT2_1 = GT2 == 1  # 1代表耻骨联合
                # GT2_2 = GT2 == 2  # 2代表胎头
                #
                # GT3 = GT3.squeeze(1)
                # GT3_0 = GT3 == 0  # 0代表其余部分
                # GT3_1 = GT3 == 1  # 1代表耻骨联合
                # GT3_2 = GT3 == 2  # 2代表胎头
                #
                # GT4 = GT4.squeeze(1)
                # GT4_0 = GT4 == 0  # 0代表其余部分
                # GT4_1 = GT4 == 1  # 1代表耻骨联合
                # GT4_2 = GT4 == 2  # 2代表胎头
                #
                # GT5 = GT5.squeeze(1)
                # GT5_0 = GT5 == 0  # 0代表其余部分
                # GT5_1 = GT5 == 1  # 1代表耻骨联合
                # GT5_2 = GT5 == 2  # 2代表胎头

                GT = torch.cat((GT_0.unsqueeze(1), GT_1.unsqueeze(1), GT_2.unsqueeze(1)), 1).int()
                # print(GT.max())
                print(cor_num)
                # yu的图片要用012 新数据用sun
                # GT = self.onehot_to_mulchannel_sun(GT)
                # GT = self.onehot_to_mulchannel(GT)


                # GT2 = torch.cat((GT2_0.unsqueeze(1), GT2_1.unsqueeze(1), GT2_2.unsqueeze(1)), 1).int()
                # GT3 = torch.cat((GT3_0.unsqueeze(1), GT3_1.unsqueeze(1), GT3_2.unsqueeze(1)), 1).int()
                # GT4 = torch.cat((GT4_0.unsqueeze(1), GT4_1.unsqueeze(1), GT4_2.unsqueeze(1)), 1).int()
                # GT5 = torch.cat((GT5_0.unsqueeze(1), GT5_1.unsqueeze(1), GT5_2.unsqueeze(1)), 1).int()

                # SR_lm, SR_seg, _, _, _, _ = self.unet(images)
                # SR_seg = self.unet(images)
                # SR_seg, SR_seg2, SR_seg3, SR_seg4, SR_seg5 = self.unet(images)
                if False:
                    SR_seg, SR_seg5 = self.unet(images)
                # SR_seg, SR_seg5 = self.unet(images)
                # SR_seg, g1, g2, g3 = self.unet(images)
                SR_seg = self.unet(images)
                # SR_seg=SR_lm
                # SR_lm = torch.sigmoid(SR_lm)
                SR_seg = torch.softmax(SR_seg, 1) > 0.5
                # SR_seg2 = torch.softmax(SR_seg2, 1) > 0.5
                # SR_seg3 = torch.softmax(SR_seg3, 1) > 0.5
                # SR_seg4 = torch.softmax(SR_seg4, 1) > 0.5
                if False:
                    SR_seg5 = torch.softmax(SR_seg5, 1) > 0.5

                SR_seg = SR_seg.int()



                Src = GT.mul(255)
                Dst = SR_seg.mul(255)
                # Dst2 = SR_seg2.mul(255)
                # Dst3 = SR_seg3.mul(255)
                # Dst4 = SR_seg4.mul(255)

                # contours, _ = cv.findContours(SR[:, :, 1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

                # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no))
                # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no))
                # temp_path = os.path.join(self.model_path, 'final_results/{}/unet_image/'.format(self.model_type))
                temp_path = os.path.join('aop_results/{}/unet_image/'.format(self.model_type)) #为每个模型进行测试并保存
                if  os.path.exists(temp_path) is False:
                    os.makedirs(temp_path)
                    print('succeed to mkdirs: {}'.format(temp_path))

                for index in range(0, len(filename)):
                # for index in range(0, len(cor_num)):
                    # filename = filename[0]
                    for i in range(0, 3): # src是GT
                        # np.expand_dims(0)
                        image_item = Src[index][i].cpu().detach().numpy().astype(np.uint8)
                        image_item = Image.fromarray(image_item, mode='L')
                        # image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i)))
                        image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i)))
                        print('-------------------std-to-real-----------------------')
                        # image_item.save('./result/pictures_DBU_1_1_1_single_0.2_upsamle_defnorm_conv/ATD-{}-src-{}.png'.format(cor_num[index], i))
                    for i in range(0, 3): # dst是测试图片
                        image_item = Dst[index][i].cpu().detach().numpy().astype(np.uint8)
                        image_item = Image.fromarray(image_item, mode='L')
                        # image_item.save(os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(cor_num[index], i)))
                        image_item.save(os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i)))


                print("generate batch {}".format(len(cor_num)))
                # print(filename)


            print("generate end ......")




    def get_max_contour(self, contours):
        max_contour = []
        for contour_item in contours:
            # print(contour_item.shape)
            if len(contour_item) > len(max_contour):
                max_contour = contour_item
        max_contour = [max_contour]  # 填充形状时必须用列表，画轮廓可用可不用
        return max_contour


    def get_area_contour(self, sect):
        _, circle_contours, hierarchy = cv2.findContours(sect, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_intersect_circle_contour = self.get_max_contour(circle_contours)
        area_contour = cv2.contourArea(max_intersect_circle_contour[0])
        return area_contour
    def generate_output_aop_sun_normal(self, mode = 'valid', dataset_no=99):
        if dataset_no == 99 and mode =='valid':
            return

        # temp_test_path = './dataset_{}/valid/valid'.format(dataset_no)
        # temp_test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # if mode == 'test':
        #     test_path = './dataset_test/test'
        # else:
        #     test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # temp_image_size = 512
        # temp_batch_size = 1
        # temp_num_workers = 1
        # self.test_loader = get_loader(image_path=test_path,
        #                               image_size=temp_image_size,
        #                               batch_size=temp_batch_size,
        #                               num_workers=temp_num_workers,
        #                               mode='test',
        #                               augmentation_prob=0.)
        # self.unet.train(False)
        # self.unet.eval()
        # 读取深度文件
        depth_Path = './csv/depth.csv'
        aopnewGT_path = './csv/aopnewGT.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        aopnewGT = np.loadtxt(aopnewGT_path, delimiter=',')
        pixel_num = self.image_size * 0.715 # why
        self.image_size_h = 384
        a_5 = 0
        a_10 = 0
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        list_aop = []
        list_l = []
        list_r = []
        list_aop_root_mse = []
        out_aop = []
        true_aop = []
        temp_path = os.path.join('aop_results/{}/unet_image/'.format(self.model_type)) #为每个模型进行测试并保存
        aop_result_path = temp_path.split('unet_image')[0]
        if  os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
            print('succeed to mkdirs: {}'.format(temp_path))
        # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no)) # 加了csv
        # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no)) # 加了csv
        dfhistory_valid_aop = pd.DataFrame(
            columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_distance",
                     "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff"])
        count = 0
        W = 1295
        H = 1026
        for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):

            for index in range(0, len(filename)):
                left_point = right_point = tangent_point = []
                standard_left_point = standard_right_point = standard_tangent_point = []
                # degth_cm = landmark_depth[cor_num[index] - 1]
                # aopnewGT_item = aopnewGT[cor_num[index] - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                # standard_left_point = (aopnewGT_item[0] * (self.image_size / W), aopnewGT_item[1] * (self.image_size_h / H)) #pixel_num
                # standard_right_point = (aopnewGT_item[2] * (self.image_size / W), aopnewGT_item[3] * (self.image_size_h / H)) #pixel_num
                # standard_tangent_point = (aopnewGT_item[4] * (self.image_size / W), aopnewGT_item[5] * (self.image_size_h / H)) #pixel_num

                temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], 0))
                # temp_path_item = os.path.join(temp_path, '623-{}-GT-{}.png'.format(filename[0], 0))
                img_background = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)
                img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGR)

                # 获得GT图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        break
                    standard_ellipse_instance = cv2.fitEllipse(max_contour[0])
                    # cv2.ellipse(img_background, standard_ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        standard_left_point, standard_right_point = get_endpoint(standard_ellipse_instance)
                    else:
                        standard_tangent_point = get_tangent_point(standard_right_point, standard_ellipse_instance)

                # 获得预测图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        flag = 1
                        break
                    ellipse_instance = cv2.fitEllipse(max_contour[0])
                    cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        left_point, right_point = get_endpoint(ellipse_instance)
                    else:
                        tangent_point = get_tangent_point(right_point, ellipse_instance)

                if flag == 1:
                    flag == 0
                    continue

                # 计算距离并保存
                # distance = pixel_mm * math.sqrt(math.pow(right_point[0] - left_point[0], 2) + math.pow(right_point[1] - left_point[1], 2))
                cv2.line(img_background, (round(left_point[0]), round(left_point[1])),
                         (round(right_point[0]), round(right_point[1])), (255, 255, 0), 1)
                # cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 2)
                cv2.line(img_background, (round(right_point[0]), round(right_point[1])),
                         (round(tangent_point[0]), round(tangent_point[1])), (0, 0, 255), 1)
                temp_path_item = os.path.join(temp_path, 'ATD-{}-real_marked_result-{}.png'.format(filename[index], 0))
                cv2.imwrite(temp_path_item, img_background)

                pic_no = 'ATD-marked_result-{}'.format(filename[index])
                left_point_pixel = math.sqrt(math.pow(left_point[0] - standard_left_point[0], 2) + math.pow(
                    left_point[1] - standard_left_point[1], 2))
                # left_point_distance = pixel_mm * left_point_pixel
                left_point_distance = left_point_pixel
                right_point_pixel = math.sqrt(math.pow(right_point[0] - standard_right_point[0], 2) + math.pow(
                    right_point[1] - standard_right_point[1], 2))
                # right_point_distance = pixel_mm * right_point_pixel
                right_point_distance =  right_point_pixel
                tangent_point_pixel = math.sqrt(math.pow(tangent_point[0] - standard_tangent_point[0], 2) + math.pow(
                    tangent_point[1] - standard_tangent_point[1], 2))
                # tangent_point_distance = pixel_mm * tangent_point_pixel
                tangent_point_distance =  tangent_point_pixel

                # 计算预测的aop角度 角度计算 a · b = |a| * |b| * cos(o)
                vector_standard_axis = (
                    standard_left_point[0] - standard_right_point[0], standard_left_point[1] - standard_right_point[1])
                vector_output_axis = (left_point[0] - right_point[0], left_point[1] - right_point[1])
                top = vector_output_axis[0] * vector_standard_axis[0] + vector_output_axis[1] * vector_standard_axis[1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                axis_angle_diff = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                vector_standard_tangent = (standard_tangent_point[0] - standard_right_point[0],
                                           standard_tangent_point[1] - standard_right_point[1])
                vector_output_tangent = (tangent_point[0] - right_point[0], tangent_point[1] - right_point[1])

                top = vector_output_axis[0] * vector_output_tangent[0] + vector_output_axis[1] * vector_output_tangent[
                    1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_output_tangent[0], 2) + math.pow(vector_output_tangent[1], 2))
                output_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                # 计算标准aop角度
                top = vector_standard_axis[0] * vector_standard_tangent[0] + vector_standard_axis[1] * \
                      vector_standard_tangent[1]
                bottom1 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_tangent[0], 2) + math.pow(vector_standard_tangent[1], 2))
                standard_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)
                # print("*********")
                print(standard_aop)
                aop_diff = math.fabs(standard_aop - output_aop)
                info = (pic_no, left_point_pixel, left_point_distance, right_point_pixel, right_point_distance,
                        tangent_point_pixel, tangent_point_distance, axis_angle_diff, aop_diff)
                # dfhistory_train.loc[len(dfhistory_train)] = info
                dfhistory_valid_aop.loc[count] = info
                count = count + 1
                dfhistory_valid_aop.to_csv(aop_result_path + 'aop_statistics.csv', index=False)
            print("generate batch {}".format(len(cor_num)))
        print("generate end ......")

    def generate_output_aop_sun_circle(self, mode = 'valid', dataset_no=99):
        if dataset_no == 99 and mode =='valid':
            return


        # self.unet.eval()
        # 读取深度文件
        depth_Path = './csv/depth.csv'
        aopnewGT_path = './csv/aopnewGT.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        aopnewGT = np.loadtxt(aopnewGT_path, delimiter=',')
        pixel_num = self.image_size * 0.715 # why
        self.image_size_h = 384
        a_5 = 0
        a_10 = 0
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        list_aop = []
        list_l = []
        list_r = []
        list_aop_root_mse = []
        out_aop = []
        true_aop = []
        temp_path = os.path.join('aop_results/{}/unet_image/'.format(self.model_type)) #为每个模型进行测试并保存
        aop_result_path = temp_path.split('unet_image')[0]
        if  os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
            print('succeed to mkdirs: {}'.format(temp_path))
        # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no)) # 加了csv
        # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no)) # 加了csv
        dfhistory_valid_aop = pd.DataFrame(
            columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_distance",
                     "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff"])
        count = 0
        W = 1295
        H = 1026
        for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):

            for index in range(0, len(filename)):
                left_point = right_point = tangent_point = []
                standard_left_point = standard_right_point = standard_tangent_point = []
                # degth_cm = landmark_depth[cor_num[index] - 1]
                # aopnewGT_item = aopnewGT[cor_num[index] - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                # standard_left_point = (aopnewGT_item[0] * (self.image_size / W), aopnewGT_item[1] * (self.image_size_h / H)) #pixel_num
                # standard_right_point = (aopnewGT_item[2] * (self.image_size / W), aopnewGT_item[3] * (self.image_size_h / H)) #pixel_num
                # standard_tangent_point = (aopnewGT_item[4] * (self.image_size / W), aopnewGT_item[5] * (self.image_size_h / H)) #pixel_num

                temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], 0))
                # temp_path_item = os.path.join(temp_path, '623-{}-GT-{}.png'.format(filename[0], 0))
                img_background = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)
                img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGR)

                # 获得GT图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        break
                    standard_ellipse_instance = cv2.fitEllipse(max_contour[0])
                    # cv2.ellipse(img_background, standard_ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        standard_left_point, standard_right_point = get_endpoint(standard_ellipse_instance)
                    else:
                        standard_tangent_point = get_tangent_point(standard_right_point, standard_ellipse_instance)

                # 获得预测图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        flag = 1
                        break
                    if i == 1:
                        ellipse_instance = cv2.fitEllipse(max_contour[0])
                    else:
                        MM = cv2.moments(max_contour[0])
                        # print(MM)
                        centroid_x = int(MM["m10"] / MM["m00"])
                        centroid_y = int(MM["m01"] / MM["m00"])
                        r = math.sqrt(cv2.contourArea(max_contour[0]) / math.pi)
                        # 在重心得到胎头圆
                        ellipse_instance = ((centroid_x, centroid_y), (2 * r, 2 * r), 0)

                    cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        left_point, right_point = get_endpoint(ellipse_instance)
                    else:
                        tangent_point = get_tangent_point(right_point, ellipse_instance)

                if flag == 1:
                    flag == 0
                    continue

                # 计算距离并保存
                # distance = pixel_mm * math.sqrt(math.pow(right_point[0] - left_point[0], 2) + math.pow(right_point[1] - left_point[1], 2))
                cv2.line(img_background, (round(left_point[0]), round(left_point[1])),
                         (round(right_point[0]), round(right_point[1])), (255, 255, 0), 1)
                # cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 2)
                cv2.line(img_background, (round(right_point[0]), round(right_point[1])),
                         (round(tangent_point[0]), round(tangent_point[1])), (0, 0, 255), 1)
                temp_path_item = os.path.join(temp_path, 'ATD-{}-real_marked_result-{}.png'.format(filename[index], 0))
                cv2.imwrite(temp_path_item, img_background)

                pic_no = 'ATD-marked_result-{}'.format(filename[index])
                left_point_pixel = math.sqrt(math.pow(left_point[0] - standard_left_point[0], 2) + math.pow(
                    left_point[1] - standard_left_point[1], 2))
                # left_point_distance = pixel_mm * left_point_pixel
                left_point_distance = left_point_pixel
                right_point_pixel = math.sqrt(math.pow(right_point[0] - standard_right_point[0], 2) + math.pow(
                    right_point[1] - standard_right_point[1], 2))
                # right_point_distance = pixel_mm * right_point_pixel
                right_point_distance =  right_point_pixel
                tangent_point_pixel = math.sqrt(math.pow(tangent_point[0] - standard_tangent_point[0], 2) + math.pow(
                    tangent_point[1] - standard_tangent_point[1], 2))
                # tangent_point_distance = pixel_mm * tangent_point_pixel
                tangent_point_distance =  tangent_point_pixel

                # 计算预测的aop角度 角度计算 a · b = |a| * |b| * cos(o)
                vector_standard_axis = (
                    standard_left_point[0] - standard_right_point[0], standard_left_point[1] - standard_right_point[1])
                vector_output_axis = (left_point[0] - right_point[0], left_point[1] - right_point[1])
                top = vector_output_axis[0] * vector_standard_axis[0] + vector_output_axis[1] * vector_standard_axis[1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                axis_angle_diff = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                vector_standard_tangent = (standard_tangent_point[0] - standard_right_point[0],
                                           standard_tangent_point[1] - standard_right_point[1])
                vector_output_tangent = (tangent_point[0] - right_point[0], tangent_point[1] - right_point[1])

                top = vector_output_axis[0] * vector_output_tangent[0] + vector_output_axis[1] * vector_output_tangent[
                    1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_output_tangent[0], 2) + math.pow(vector_output_tangent[1], 2))
                output_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                # 计算标准aop角度
                top = vector_standard_axis[0] * vector_standard_tangent[0] + vector_standard_axis[1] * \
                      vector_standard_tangent[1]
                bottom1 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_tangent[0], 2) + math.pow(vector_standard_tangent[1], 2))
                standard_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)
                # print("*********")
                print(standard_aop)
                aop_diff = math.fabs(standard_aop - output_aop)
                info = (pic_no, left_point_pixel, left_point_distance, right_point_pixel, right_point_distance,
                        tangent_point_pixel, tangent_point_distance, axis_angle_diff, aop_diff)
                # dfhistory_train.loc[len(dfhistory_train)] = info
                dfhistory_valid_aop.loc[count] = info
                count = count + 1
                dfhistory_valid_aop.to_csv(aop_result_path + 'aop_statistics.csv', index=False)
            print("generate batch {}".format(len(cor_num)))
        print("generate end ......")

    def generate_output_aop_sun_both(self, mode = 'valid', dataset_no=99):
        if dataset_no == 99 and mode =='valid':
            return


        # self.unet.eval()
        # 读取深度文件
        depth_Path = './csv/depth.csv'
        aopnewGT_path = './csv/aopnewGT.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        aopnewGT = np.loadtxt(aopnewGT_path, delimiter=',')
        pixel_num = self.image_size * 0.715 # why
        self.image_size_h = 384
        a_5 = 0
        a_10 = 0
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        list_aop = []
        list_l = []
        list_r = []
        list_aop_root_mse = []
        out_aop = []
        true_aop = []
        temp_path = os.path.join('aop_results/{}/unet_image/'.format(self.model_type)) #为每个模型进行测试并保存
        aop_result_path = temp_path.split('unet_image')[0]
        if  os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
            print('succeed to mkdirs: {}'.format(temp_path))
        # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no)) # 加了csv
        # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no)) # 加了csv
        dfhistory_valid_aop = pd.DataFrame(
            columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_distance",
                     "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff", "ratio_circle",
                     'ratio_ellipse'])
        count = 0
        W = 1295
        H = 1026
        for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):

            for index in range(0, len(filename)):
                left_point = right_point = tangent_point = []
                standard_left_point = standard_right_point = standard_tangent_point = []
                # degth_cm = landmark_depth[cor_num[index] - 1]
                # aopnewGT_item = aopnewGT[cor_num[index] - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                # standard_left_point = (aopnewGT_item[0] * (self.image_size / W), aopnewGT_item[1] * (self.image_size_h / H)) #pixel_num
                # standard_right_point = (aopnewGT_item[2] * (self.image_size / W), aopnewGT_item[3] * (self.image_size_h / H)) #pixel_num
                # standard_tangent_point = (aopnewGT_item[4] * (self.image_size / W), aopnewGT_item[5] * (self.image_size_h / H)) #pixel_num

                temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], 0))
                # temp_path_item = os.path.join(temp_path, '623-{}-GT-{}.png'.format(filename[0], 0))
                img_background = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)
                img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGR)
                ratio_circle = 0
                ratio_ellipse = 0

                # 获得GT图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        break
                    standard_ellipse_instance = cv2.fitEllipse(max_contour[0])
                    # cv2.ellipse(img_background, standard_ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        standard_left_point, standard_right_point = get_endpoint(standard_ellipse_instance)
                    else:
                        standard_tangent_point = get_tangent_point(standard_right_point, standard_ellipse_instance)

                # 获得预测图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        flag = 1
                        break
                    if i == 1:
                        ellipse_instance = cv2.fitEllipse(max_contour[0])
                    else:
                        MM = cv2.moments(max_contour[0])
                        # print(MM)
                        centroid_x = int(MM["m10"] / MM["m00"])
                        centroid_y = int(MM["m01"] / MM["m00"])
                        r = math.sqrt(cv2.contourArea(max_contour[0]) / math.pi)
                        # 在重心得到胎头圆
                        circle_instance_choice = ((centroid_x, centroid_y), (2 * r, 2 * r), 0)
                        ellipse_instance_choice =  cv2.fitEllipse(max_contour[0])

                        #计算重心圆和胎头的相交区域
                        temp_img_circle = np.zeros((384, 512, 1), np.uint8)
                        cv2.ellipse(temp_img_circle, circle_instance_choice, (255, 255, 255), thickness=-1)
                        head_img = img_item.copy()
                        intersect_circle = cv2.bitwise_and(temp_img_circle, head_img)
                        union_circle = cv2.bitwise_or(temp_img_circle, head_img)

                        #计算椭圆和胎头的相交区域
                        temp_img_ellipse = np.zeros((384, 512, 1), np.uint8)
                        cv2.ellipse(temp_img_ellipse, ellipse_instance_choice, (255, 255, 255), thickness=-1)
                        intersect_ellipse = cv2.bitwise_and(temp_img_ellipse, head_img)
                        union_ellipse = cv2.bitwise_or(temp_img_ellipse, head_img)

                        #计算重心圆和分割区域交集的面积 先获取轮廓然后调用函数
                        # _, circle_contours, hierarchy = cv2.findContours(intersect_circle, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        #  #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        # max_intersect_circle_contour = self.get_max_contour(circle_contours)
                        # area_circle = cv2.contourArea(max_intersect_circle_contour[0])
                        circle_area_inter = self.get_area_contour(intersect_circle)
                        circle_area_union = self.get_area_contour(union_circle)
                        ratio_circle = circle_area_inter * 1.0 / circle_area_union

                        #计算椭圆和分割区域交集的面积 先获取轮廓然后调用函数
                        # _, ellipse_contours, hierarchy = cv2.findContours(intersect_ellipse, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        # #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        # max_intersect_ellipse_contour = self.get_max_contour(ellipse_contours)
                        # area_ellipse = cv2.contourArea(max_intersect_ellipse_contour[0])
                        ellipse_area_inter = self.get_area_contour(intersect_ellipse)
                        ellipse_area_union = self.get_area_contour(union_ellipse)
                        ratio_ellipse = ellipse_area_inter * 1.0 / ellipse_area_union

                        if ratio_ellipse > ratio_circle:
                            ellipse_instance = ellipse_instance_choice
                        else:
                            ellipse_instance = circle_instance_choice

                    cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        left_point, right_point = get_endpoint(ellipse_instance)
                    else:
                        tangent_point = get_tangent_point(right_point, ellipse_instance)

                if flag == 1:
                    flag == 0
                    continue

                # 计算距离并保存
                # distance = pixel_mm * math.sqrt(math.pow(right_point[0] - left_point[0], 2) + math.pow(right_point[1] - left_point[1], 2))
                cv2.line(img_background, (round(left_point[0]), round(left_point[1])),
                         (round(right_point[0]), round(right_point[1])), (255, 255, 0), 1)
                # cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 2)
                cv2.line(img_background, (round(right_point[0]), round(right_point[1])),
                         (round(tangent_point[0]), round(tangent_point[1])), (0, 0, 255), 1)
                temp_path_item = os.path.join(temp_path, 'ATD-{}-real_marked_result-{}.png'.format(filename[index], 0))
                cv2.imwrite(temp_path_item, img_background)

                pic_no = 'ATD-marked_result-{}'.format(filename[index])
                left_point_pixel = math.sqrt(math.pow(left_point[0] - standard_left_point[0], 2) + math.pow(
                    left_point[1] - standard_left_point[1], 2))
                # left_point_distance = pixel_mm * left_point_pixel
                left_point_distance = left_point_pixel
                right_point_pixel = math.sqrt(math.pow(right_point[0] - standard_right_point[0], 2) + math.pow(
                    right_point[1] - standard_right_point[1], 2))
                # right_point_distance = pixel_mm * right_point_pixel
                right_point_distance =  right_point_pixel
                tangent_point_pixel = math.sqrt(math.pow(tangent_point[0] - standard_tangent_point[0], 2) + math.pow(
                    tangent_point[1] - standard_tangent_point[1], 2))
                # tangent_point_distance = pixel_mm * tangent_point_pixel
                tangent_point_distance =  tangent_point_pixel

                # 计算预测的aop角度 角度计算 a · b = |a| * |b| * cos(o)
                vector_standard_axis = (
                    standard_left_point[0] - standard_right_point[0], standard_left_point[1] - standard_right_point[1])
                vector_output_axis = (left_point[0] - right_point[0], left_point[1] - right_point[1])
                top = vector_output_axis[0] * vector_standard_axis[0] + vector_output_axis[1] * vector_standard_axis[1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                axis_angle_diff = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                vector_standard_tangent = (standard_tangent_point[0] - standard_right_point[0],
                                           standard_tangent_point[1] - standard_right_point[1])
                vector_output_tangent = (tangent_point[0] - right_point[0], tangent_point[1] - right_point[1])

                top = vector_output_axis[0] * vector_output_tangent[0] + vector_output_axis[1] * vector_output_tangent[
                    1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_output_tangent[0], 2) + math.pow(vector_output_tangent[1], 2))
                output_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                # 计算标准aop角度
                top = vector_standard_axis[0] * vector_standard_tangent[0] + vector_standard_axis[1] * \
                      vector_standard_tangent[1]
                bottom1 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_tangent[0], 2) + math.pow(vector_standard_tangent[1], 2))
                standard_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)
                # print("*********")
                print(standard_aop)
                aop_diff = math.fabs(standard_aop - output_aop)
                info = (pic_no, left_point_pixel, left_point_distance, right_point_pixel, right_point_distance,
                        tangent_point_pixel, tangent_point_distance, axis_angle_diff, aop_diff, ratio_circle,
                        ratio_ellipse)
                # dfhistory_train.loc[len(dfhistory_train)] = info
                dfhistory_valid_aop.loc[count] = info
                count = count + 1
                dfhistory_valid_aop.to_csv(aop_result_path + 'aop_statistics.csv', index=False)
            print("generate batch {}".format(len(cor_num)))
        print("generate end ......")

    def getContours(self, img_src):
        _, contours, hierarchy = cv2.findContours(img_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_contour = []
        for contour_item in contours:
            if len(contour_item) > len(max_contour):
                max_contour = contour_item
        max_contour_head = [max_contour]
        return max_contour_head

    def generate_output_aop_sun_clip(self, mode = 'valid', dataset_no=99):
        if dataset_no == 99 and mode =='valid':
            return

        kk = 5
        k = 0.005 * kk
        head_ellipse_area_range = (1-k, 1+k)
        head_ellipse_girth_range = (1-k, 1+k)
        head_counts = 0
        # self.unet.eval()
        # 读取深度文件
        depth_Path = './csv/depth.csv'
        aopnewGT_path = './csv/aopnewGT.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        aopnewGT = np.loadtxt(aopnewGT_path, delimiter=',')
        pixel_num = self.image_size * 0.715 # why
        self.image_size_h = 384
        a_5 = 0
        a_10 = 0
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        list_aop = []
        list_l = []
        list_r = []
        list_aop_root_mse = []
        out_aop = []
        true_aop = []
        temp_path = os.path.join('aop_results/{}/unet_image/'.format(self.model_type)) #为每个模型进行测试并保存
        aop_result_path = temp_path.split('unet_image')[0]
        if  os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
            print('succeed to mkdirs: {}'.format(temp_path))
        # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no)) # 加了csv
        # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no)) # 加了csv
        dfhistory_valid_aop = pd.DataFrame(
            columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_distance",
                     "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff", "ratio_circle",
                     'ratio_ellipse'])
        count = 0
        W = 1295
        H = 1026
        for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):

            for index in range(0, len(filename)):
                left_point = right_point = tangent_point = []
                standard_left_point = standard_right_point = standard_tangent_point = []
                # degth_cm = landmark_depth[cor_num[index] - 1]
                # aopnewGT_item = aopnewGT[cor_num[index] - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                # standard_left_point = (aopnewGT_item[0] * (self.image_size / W), aopnewGT_item[1] * (self.image_size_h / H)) #pixel_num
                # standard_right_point = (aopnewGT_item[2] * (self.image_size / W), aopnewGT_item[3] * (self.image_size_h / H)) #pixel_num
                # standard_tangent_point = (aopnewGT_item[4] * (self.image_size / W), aopnewGT_item[5] * (self.image_size_h / H)) #pixel_num

                temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], 0))
                # temp_path_item = os.path.join(temp_path, '623-{}-GT-{}.png'.format(filename[0], 0))
                img_background = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)
                img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGR)
                ratio_circle = 0
                ratio_ellipse = 0

                # 获得GT图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        break
                    standard_ellipse_instance = cv2.fitEllipse(max_contour[0])
                    # cv2.ellipse(img_background, standard_ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        standard_left_point, standard_right_point = get_endpoint(standard_ellipse_instance)
                    else:
                        standard_tangent_point = get_tangent_point(standard_right_point, standard_ellipse_instance)

                # 获得预测图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        flag = 1
                        break
                    if i == 1:
                        ellipse_instance = cv2.fitEllipse(max_contour[0])
                    else:
                        head_contours = self.getContours(img_item)
                        head_ellipse = cv2.fitEllipse(head_contours[0])
                        #循环
                        remain_head_img = img_item.copy()
                        remain_head_ellipse = head_ellipse
                        remain_head_contours = head_contours
                        # 循环减去区域 得到胎头椭圆,最多3次?
                        head_clip = 1

                        for j in range(0, head_clip):
                            # head_counts = head_counts + 1
                            remain_head_contours = self.getContours(remain_head_img)
                            remain_head_ellipse = cv2.fitEllipse(remain_head_contours[0])

                            # 计算重心
                            MM = cv2.moments(remain_head_contours[0])
                            # print(MM)
                            centroid_x = int(MM["m10"] / MM["m00"])
                            centroid_y = int(MM["m01"] / MM["m00"])
                            #移动椭圆
                            remain_head_ellipse = ((centroid_x,centroid_y), remain_head_ellipse[1], remain_head_ellipse[2])
                            # 裁剪其余部分， 留下胎头交集
                            temp_img = np.zeros((384, 512, 1), np.uint8)
                            cv2.ellipse(temp_img, remain_head_ellipse, (255, 255, 255), thickness=-1)
                            remain_head_img = cv2.bitwise_and(temp_img, remain_head_img)
                            # 新图像写入磁盘
                            # path = root[:-5] + '/clip_' + str(i + 1) + '/head_' + str(pic_no) + '.jpg'
                            # if not os.path.exists(root[:-5] + '/clip_' + str(i + 1) + '/'):
                            #     os.mkdir(root[:-5] + '/clip_' + str(i + 1) + '/')
                            # if not os.path.exists(path):
                            #     cv2.imwrite(path, remain_head_img)
                            # 判断是否提前停止
                            # head_flag_girth = False
                            # head_flag_area = False
                            # # sp_ellipse = cv2.fitEllipse(sp_contour)
                            # head_ellipse_area = math.pi * remain_head_ellipse[1][0] * remain_head_ellipse[1][1] / 4
                            # a = max(remain_head_ellipse[1][0], remain_head_ellipse[1][1]) / 2
                            # b = min(remain_head_ellipse[1][0], remain_head_ellipse[1][1]) / 2
                            # head_ellipse_girth = 2 * math.pi * b + 4 * (a - b)
                            # tt_contours = self.getContours(remain_head_img)
                            # head_real_area = cv2.contourArea(tt_contours[0])
                            # head_real_girth = cv2.arcLength(tt_contours[0], True)
                            #
                            # head_area_ratio = head_ellipse_area / head_real_area
                            # head_girth_ratio = head_ellipse_girth / head_real_girth
                            # if head_area_ratio >= head_ellipse_area_range[0] and head_area_ratio <= head_ellipse_area_range[1]:
                            #     head_flag_area = True
                            # if head_girth_ratio >= head_ellipse_girth_range[0] and head_girth_ratio <= head_ellipse_girth_range[1]:
                            #     head_flag_girth = True
                            # if head_flag_area == True and head_flag_girth == True:
                            #     break
                                # 获得新轮廓，拟合新椭圆
                                ##remain_head_contours = getContours(remain_head_img)
                                ##remain_head_ellipse = cv2.fitEllipse(remain_head_contours[0])

                                # head_ellipse = cv2.fitEllipse(remain_head_contours[0])
                                # # 在重心得到胎头椭圆
                                # head_ellipse = ((centroid_x, centroid_y), head_ellipse[1], head_ellipse[2])
                                # # 融合区域求交集
                                # cv2.ellipse(temp_img, head_ellipse, (255, 255, 255), thickness=-1)
                                # remain_head_img = cv2.bitwise_and(temp_img, remain_head_img)

                                # 判断是否可以提前终止
                                # if True:
                                #     break
                        ellipse_instance = remain_head_ellipse


                    cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        left_point, right_point = get_endpoint(ellipse_instance)
                    else:
                        tangent_point = get_tangent_point(right_point, ellipse_instance)

                # if flag == 1:
                #     flag == 0
                #     continue
                if len(tangent_point) == 0:
                    continue
                # 计算距离并保存
                # distance = pixel_mm * math.sqrt(math.pow(right_point[0] - left_point[0], 2) + math.pow(right_point[1] - left_point[1], 2))
                cv2.line(img_background, (round(left_point[0]), round(left_point[1])),
                         (round(right_point[0]), round(right_point[1])), (255, 255, 0), 1)
                # cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 2)
                cv2.line(img_background, (round(right_point[0]), round(right_point[1])),
                         (round(tangent_point[0]), round(tangent_point[1])), (0, 0, 255), 1)
                temp_path_item = os.path.join(temp_path, 'ATD-{}-real_marked_result-{}.png'.format(filename[index], 0))
                cv2.imwrite(temp_path_item, img_background)

                pic_no = 'ATD-marked_result-{}'.format(filename[index])
                left_point_pixel = math.sqrt(math.pow(left_point[0] - standard_left_point[0], 2) + math.pow(
                    left_point[1] - standard_left_point[1], 2))
                # left_point_distance = pixel_mm * left_point_pixel
                left_point_distance = left_point_pixel
                right_point_pixel = math.sqrt(math.pow(right_point[0] - standard_right_point[0], 2) + math.pow(
                    right_point[1] - standard_right_point[1], 2))
                # right_point_distance = pixel_mm * right_point_pixel
                right_point_distance =  right_point_pixel
                tangent_point_pixel = math.sqrt(math.pow(tangent_point[0] - standard_tangent_point[0], 2) + math.pow(
                    tangent_point[1] - standard_tangent_point[1], 2))
                # tangent_point_distance = pixel_mm * tangent_point_pixel
                tangent_point_distance =  tangent_point_pixel

                # 计算预测的aop角度 角度计算 a · b = |a| * |b| * cos(o)
                vector_standard_axis = (
                    standard_left_point[0] - standard_right_point[0], standard_left_point[1] - standard_right_point[1])
                vector_output_axis = (left_point[0] - right_point[0], left_point[1] - right_point[1])
                top = vector_output_axis[0] * vector_standard_axis[0] + vector_output_axis[1] * vector_standard_axis[1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                axis_angle_diff = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                vector_standard_tangent = (standard_tangent_point[0] - standard_right_point[0],
                                           standard_tangent_point[1] - standard_right_point[1])
                vector_output_tangent = (tangent_point[0] - right_point[0], tangent_point[1] - right_point[1])

                top = vector_output_axis[0] * vector_output_tangent[0] + vector_output_axis[1] * vector_output_tangent[
                    1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_output_tangent[0], 2) + math.pow(vector_output_tangent[1], 2))
                output_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                # 计算标准aop角度
                top = vector_standard_axis[0] * vector_standard_tangent[0] + vector_standard_axis[1] * \
                      vector_standard_tangent[1]
                bottom1 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_tangent[0], 2) + math.pow(vector_standard_tangent[1], 2))
                standard_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)
                # print("*********")
                print(standard_aop)
                aop_diff = math.fabs(standard_aop - output_aop)
                info = (pic_no, left_point_pixel, left_point_distance, right_point_pixel, right_point_distance,
                        tangent_point_pixel, tangent_point_distance, axis_angle_diff, aop_diff, ratio_circle,
                        ratio_ellipse)
                # dfhistory_train.loc[len(dfhistory_train)] = info
                dfhistory_valid_aop.loc[count] = info
                count = count + 1
                dfhistory_valid_aop.to_csv(aop_result_path + 'aop_statistics.csv', index=False)
            print("generate batch {}".format(len(cor_num)))
        print("generate end ......")

    def generate_output_aop_sun_hull(self, mode = 'valid', dataset_no=99):
        if dataset_no == 99 and mode =='valid':
            return

        # temp_test_path = './dataset_{}/valid/valid'.format(dataset_no)
        # temp_test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # if mode == 'test':
        #     test_path = './dataset_test/test'
        # else:
        #     test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # temp_image_size = 512
        # temp_batch_size = 1
        # temp_num_workers = 1
        # self.test_loader = get_loader(image_path=test_path,
        #                               image_size=temp_image_size,
        #                               batch_size=temp_batch_size,
        #                               num_workers=temp_num_workers,
        #                               mode='test',
        #                               augmentation_prob=0.)
        # self.unet.train(False)
        # self.unet.eval()
        # 读取深度文件
        depth_Path = './csv/depth.csv'
        aopnewGT_path = './csv/aopnewGT.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        aopnewGT = np.loadtxt(aopnewGT_path, delimiter=',')
        pixel_num = self.image_size * 0.715 # why
        self.image_size_h = 384
        a_5 = 0
        a_10 = 0
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        list_aop = []
        list_l = []
        list_r = []
        list_aop_root_mse = []
        out_aop = []
        true_aop = []
        temp_path = os.path.join('aop_results/{}/unet_image/'.format(self.model_type)) #为每个模型进行测试并保存
        aop_result_path = temp_path.split('unet_image')[0]
        if  os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
            print('succeed to mkdirs: {}'.format(temp_path))
        # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no)) # 加了csv
        # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no)) # 加了csv
        dfhistory_valid_aop = pd.DataFrame(
            columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_distance",
                     "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff"])
        count = 0
        W = 1295
        H = 1026
        for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):

            for index in range(0, len(filename)):
                left_point = right_point = tangent_point = []
                standard_left_point = standard_right_point = standard_tangent_point = []
                # degth_cm = landmark_depth[cor_num[index] - 1]
                # aopnewGT_item = aopnewGT[cor_num[index] - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                # standard_left_point = (aopnewGT_item[0] * (self.image_size / W), aopnewGT_item[1] * (self.image_size_h / H)) #pixel_num
                # standard_right_point = (aopnewGT_item[2] * (self.image_size / W), aopnewGT_item[3] * (self.image_size_h / H)) #pixel_num
                # standard_tangent_point = (aopnewGT_item[4] * (self.image_size / W), aopnewGT_item[5] * (self.image_size_h / H)) #pixel_num

                temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], 0))
                # temp_path_item = os.path.join(temp_path, '623-{}-GT-{}.png'.format(filename[0], 0))
                img_background = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)
                img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGR)

                # 获得GT图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        break
                    standard_ellipse_instance = cv2.fitEllipse(max_contour[0])
                    # cv2.ellipse(img_background, standard_ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        standard_left_point, standard_right_point = get_endpoint(standard_ellipse_instance)
                    else:
                        standard_tangent_point = get_tangent_point(standard_right_point, standard_ellipse_instance)

                # 获得预测图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        flag = 1
                        break
                    hull =  cv2.convexHull(max_contour[0])
                    ellipse_instance = cv2.fitEllipse(hull)
                    cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        left_point, right_point = get_endpoint(ellipse_instance)
                    else:
                        tangent_point = get_tangent_point(right_point, ellipse_instance)

                if flag == 1:
                    flag == 0
                    continue

                # 计算距离并保存
                # distance = pixel_mm * math.sqrt(math.pow(right_point[0] - left_point[0], 2) + math.pow(right_point[1] - left_point[1], 2))
                cv2.line(img_background, (round(left_point[0]), round(left_point[1])),
                         (round(right_point[0]), round(right_point[1])), (255, 255, 0), 1)
                # cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 2)
                cv2.line(img_background, (round(right_point[0]), round(right_point[1])),
                         (round(tangent_point[0]), round(tangent_point[1])), (0, 0, 255), 1)
                temp_path_item = os.path.join(temp_path, 'ATD-{}-real_marked_result-{}.png'.format(filename[index], 0))
                cv2.imwrite(temp_path_item, img_background)

                pic_no = 'ATD-marked_result-{}'.format(filename[index])
                left_point_pixel = math.sqrt(math.pow(left_point[0] - standard_left_point[0], 2) + math.pow(
                    left_point[1] - standard_left_point[1], 2))
                # left_point_distance = pixel_mm * left_point_pixel
                left_point_distance = left_point_pixel
                right_point_pixel = math.sqrt(math.pow(right_point[0] - standard_right_point[0], 2) + math.pow(
                    right_point[1] - standard_right_point[1], 2))
                # right_point_distance = pixel_mm * right_point_pixel
                right_point_distance =  right_point_pixel
                tangent_point_pixel = math.sqrt(math.pow(tangent_point[0] - standard_tangent_point[0], 2) + math.pow(
                    tangent_point[1] - standard_tangent_point[1], 2))
                # tangent_point_distance = pixel_mm * tangent_point_pixel
                tangent_point_distance =  tangent_point_pixel

                # 计算预测的aop角度 角度计算 a · b = |a| * |b| * cos(o)
                vector_standard_axis = (
                    standard_left_point[0] - standard_right_point[0], standard_left_point[1] - standard_right_point[1])
                vector_output_axis = (left_point[0] - right_point[0], left_point[1] - right_point[1])
                top = vector_output_axis[0] * vector_standard_axis[0] + vector_output_axis[1] * vector_standard_axis[1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                axis_angle_diff = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                vector_standard_tangent = (standard_tangent_point[0] - standard_right_point[0],
                                           standard_tangent_point[1] - standard_right_point[1])
                vector_output_tangent = (tangent_point[0] - right_point[0], tangent_point[1] - right_point[1])

                top = vector_output_axis[0] * vector_output_tangent[0] + vector_output_axis[1] * vector_output_tangent[
                    1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_output_tangent[0], 2) + math.pow(vector_output_tangent[1], 2))
                output_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                # 计算标准aop角度
                top = vector_standard_axis[0] * vector_standard_tangent[0] + vector_standard_axis[1] * \
                      vector_standard_tangent[1]
                bottom1 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_tangent[0], 2) + math.pow(vector_standard_tangent[1], 2))
                standard_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)
                # print("*********")
                print(standard_aop)
                aop_diff = math.fabs(standard_aop - output_aop)
                info = (pic_no, left_point_pixel, left_point_distance, right_point_pixel, right_point_distance,
                        tangent_point_pixel, tangent_point_distance, axis_angle_diff, aop_diff)
                # dfhistory_train.loc[len(dfhistory_train)] = info
                dfhistory_valid_aop.loc[count] = info
                count = count + 1
                dfhistory_valid_aop.to_csv(aop_result_path + 'aop_statistics.csv', index=False)
            print("generate batch {}".format(len(cor_num)))
        print("generate end ......")



    def generate_output_aop_75(self, mode = 'valid', dataset_no=99):
        if dataset_no == 99 and mode =='valid':
            return

        # temp_test_path = './dataset_{}/valid/valid'.format(dataset_no)
        # temp_test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        if mode == 'test':
            test_path = './dataset_test/test'
        else:
            test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        temp_image_size = 512
        temp_batch_size = 1
        temp_num_workers = 1
        self.test_loader = get_loader(image_path=test_path,
                                      image_size=temp_image_size,
                                      batch_size=temp_batch_size,
                                      num_workers=temp_num_workers,
                                      mode='test',
                                      augmentation_prob=0.)
        self.unet.train(False)
        self.unet.eval()
        # 读取深度文件
        depth_Path = './csv/depth.csv'
        aopnewGT_path = './csv/aopnewGT.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        aopnewGT = np.loadtxt(aopnewGT_path, delimiter=',')
        pixel_num = self.image_size * 0.715 # why
        self.image_size_h = 384
        a_5 = 0
        a_10 = 0
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        list_aop = []
        list_l = []
        list_r = []
        list_aop_root_mse = []
        out_aop = []
        true_aop = []
        temp_path = os.path.join('aop_results/{}/unet_image/'.format(self.model_type)) #为每个模型进行测试并保存
        aop_result_path = temp_path.split('unet')[0]
        if  os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
            print('succeed to mkdirs: {}'.format(temp_path))
        # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no)) # 加了csv
        # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no)) # 加了csv
        dfhistory_valid_aop = pd.DataFrame(
            columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_distance",
                     "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff"])
        count = 0
        W = 1295
        H = 1026
        for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):

            for index in range(0, len(cor_num)):
                left_point = right_point = tangent_point = []
                standard_left_point = standard_right_point = standard_tangent_point = []
                degth_cm = landmark_depth[cor_num[index] - 1]
                aopnewGT_item = aopnewGT[cor_num[index] - 1]
                pixel_mm = degth_cm * 10 / pixel_num
                standard_left_point = (aopnewGT_item[0] * (self.image_size / W), aopnewGT_item[1] * (self.image_size_h / H)) #pixel_num
                standard_right_point = (aopnewGT_item[2] * (self.image_size / W), aopnewGT_item[3] * (self.image_size_h / H)) #pixel_num
                standard_tangent_point = (aopnewGT_item[4] * (self.image_size / W), aopnewGT_item[5] * (self.image_size_h / H)) #pixel_num

                temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(cor_num[index], 0))
                # temp_path_item = os.path.join(temp_path, '623-{}-GT-{}.png'.format(filename[0], 0))
                img_background = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)
                img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGR)

                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(cor_num[index], i))
                    print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        flag = 1
                        break
                    ellipse_instance = cv2.fitEllipse(max_contour[0])
                    cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        left_point, right_point = get_endpoint(ellipse_instance)
                    else:
                        tangent_point = get_tangent_point(right_point, ellipse_instance)

                if flag == 1:
                    flag == 0
                    continue

                # 计算距离并保存
                # distance = pixel_mm * math.sqrt(math.pow(right_point[0] - left_point[0], 2) + math.pow(right_point[1] - left_point[1], 2))
                cv2.line(img_background, (round(left_point[0]), round(left_point[1])),
                         (round(right_point[0]), round(right_point[1])), (255, 255, 0), 1)
                # cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 2)
                cv2.line(img_background, (round(right_point[0]), round(right_point[1])),
                         (round(tangent_point[0]), round(tangent_point[1])), (0, 0, 255), 1)
                temp_path_item = os.path.join(temp_path, 'ATD-{}-real_marked_result-{}.png'.format(cor_num[index], 0))
                cv2.imwrite(temp_path_item, img_background)

                pic_no = 'ATD-marked_result-{}'.format(cor_num[index])
                left_point_pixel = math.sqrt(math.pow(left_point[0] - standard_left_point[0], 2) + math.pow(
                    left_point[1] - standard_left_point[1], 2))
                left_point_distance = pixel_mm * left_point_pixel
                right_point_pixel = math.sqrt(math.pow(right_point[0] - standard_right_point[0], 2) + math.pow(
                    right_point[1] - standard_right_point[1], 2))
                right_point_distance = pixel_mm * right_point_pixel
                tangent_point_pixel = math.sqrt(math.pow(tangent_point[0] - standard_tangent_point[0], 2) + math.pow(
                    tangent_point[1] - standard_tangent_point[1], 2))
                tangent_point_distance = pixel_mm * tangent_point_pixel

                # 计算预测的aop角度 角度计算 a · b = |a| * |b| * cos(o)
                vector_standard_axis = (
                    standard_left_point[0] - standard_right_point[0], standard_left_point[1] - standard_right_point[1])
                vector_output_axis = (left_point[0] - right_point[0], left_point[1] - right_point[1])
                top = vector_output_axis[0] * vector_standard_axis[0] + vector_output_axis[1] * vector_standard_axis[1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                axis_angle_diff = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                vector_standard_tangent = (standard_tangent_point[0] - standard_right_point[0],
                                           standard_tangent_point[1] - standard_right_point[1])
                vector_output_tangent = (tangent_point[0] - right_point[0], tangent_point[1] - right_point[1])

                top = vector_output_axis[0] * vector_output_tangent[0] + vector_output_axis[1] * vector_output_tangent[
                    1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_output_tangent[0], 2) + math.pow(vector_output_tangent[1], 2))
                output_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)

                # 计算标准aop角度
                top = vector_standard_axis[0] * vector_standard_tangent[0] + vector_standard_axis[1] * \
                      vector_standard_tangent[1]
                bottom1 = math.sqrt(math.pow(vector_standard_axis[0], 2) + math.pow(vector_standard_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_standard_tangent[0], 2) + math.pow(vector_standard_tangent[1], 2))
                standard_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)
                # print("*********")
                print(standard_aop)
                aop_diff = math.fabs(standard_aop - output_aop)
                info = (pic_no, left_point_pixel, left_point_distance, right_point_pixel, right_point_distance,
                        tangent_point_pixel, tangent_point_distance, axis_angle_diff, aop_diff)
                # dfhistory_train.loc[len(dfhistory_train)] = info
                dfhistory_valid_aop.loc[count] = info
                count = count + 1
                dfhistory_valid_aop.to_csv(aop_result_path + 'aop_statistics.csv', index=False)
            print("generate batch {}".format(len(cor_num)))
        print("generate end ......")

    def generate_result_test_sun(self):

        result_history = pd.DataFrame(
            columns=["dataset_no", "left_pixel_mean", "left_pixel_median", "left_pixel_var", "left_distance_mean", "left_distance_median", "left_distance_var",
                     "right_pixel_mean", "right_pixel_median", "right_pixel_var", "right_distance_mean", "right_distance_median", "right_distance_var",
                     "axis_diff_mean", "axis_diff_median", "axis_diff_var", "aop_diff_mean", "aop_diff_median", "aop_diff_var"]
        )
        count = 0
        for dataset_no in range(1):# final_results/new_0.715_to_0.75/AG_DBU_old/dataset{}/
            # csv_path = os.path.join(self.model_path, 'final_results/dataset{}/aop_history.csv'.format(dataset_no))
            # csv_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/aop_history.csv')

            # aop_statistics_result_path = 'aop_results/{}/aop_statistics_both.csv'.format(self.model_type) #为每个模型进行测试并保存
            aop_statistics_result_path = 'aop_results/{}/aop_statistics.csv'.format(self.model_type) #为每个模型进行测试并保存
            item_history = pd.read_csv(aop_statistics_result_path)
            list_pixel = item_history['left_point_pixel']
            left_pixel_mean, left_pixel_median, left_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['left_point_distance']
            left_distance_mean, left_distance_median, left_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_pixel']
            right_pixel_mean, right_pixel_median, right_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_distance']
            right_distance_mean, right_distance_median, right_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['axis_angle_diff']
            axis_diff_mean, axis_diff_median, axis_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['aop_diff']
            aop_diff_mean, aop_diff_median, aop_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))

            info = (dataset_no, left_pixel_mean, left_pixel_median, left_pixel_var, left_distance_mean, left_distance_median, left_distance_var,
                    right_pixel_mean, right_pixel_median, right_pixel_var, right_distance_mean, right_distance_median, right_distance_var,
                    axis_diff_mean, axis_diff_median, axis_diff_var, aop_diff_mean, aop_diff_median, aop_diff_var)
            result_history.loc[count] = info
            count = count + 1

        aop_analysis_path = aop_statistics_result_path.split('aop_statistics')[0]
        # result_history.to_csv(aop_analysis_path + 'aop_analysis_both.csv', index=False)
        result_history.to_csv(aop_analysis_path + 'aop_analysis.csv', index=False)
        print("genenrate result_history success ......")

    def generate_output_pic_g123(self):

        # set3_bt2_md-unet-Final_U_YS_Net_16-21-0.0001-139-0.3000-512.pkl
        # set3_bt2_md-unet-Final_U_YS_Net_16-14-0.0001-139-0.3000-512.pkl


        #加载模型直接测试
        # set3_bt2_md-unet-MBU_YS_Net_16-4-0.0001-139-0.3000-512.pkl
        # unet_path = os.path.join(self.model_path, 'final_models/dataset1/set3_bt2_md-unet-DBU_YS_Net_16-32-0.0001-139-0.3000-512.pkl')
        #### /models/templmodels/set3 bt2 md-unet-fcnlresnet50-49-0.0001-139-0.30vm0-512.pkl
        # unet_path = './models/temp_models/set3_bt2_md-unet-fcn_resnet50-49-0.0001-139-0.3000-512.pkl'
        unet_path = 'hh./models/temp_models/set3_bt2_md-unet-U_Net-299-0.9078-0.0000-139-0.3000-51.pklhhh' #3chu
        #unet_path = './result/model_weight_4000/U_Net_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/My_Unet_has_multi_has_shape_no_attention_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/AttU_Net_0.0001_1_dc0.9384.pth'

        # unet_path = './result/model_weight_4000/Multi_U_Net_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/My_Unet_no_multi_has_shape_no_attention_0.0001_1.pth'
        unet_path = './result/model_weight_4000/My_Unet_has_multi_has_shape_no_attention_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/AG_DBU_YS_Net_16_0.0001_1.pth'

        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')
            return

        # test读取测试集313张图片 否则读取5折中的验证集合的图片
        # if mode == 'test':
        #     test_path = './dataset_test/test'
        # else:
        #     test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # temp_test_path = './dataset_{}/valid/valid'.format(dataset_no)
        # temp_test_path = './dataset_5_fold/sun_dataset_{}/valid2/valid'.format(dataset_no)
        # temp_test_path = './dataset_test/test'.format(dataset_no)
        temp_image_size = 512
        temp_batch_size = 1
        temp_num_workers = 1
        # self.test_loader = get_loader(image_path=test_path,
        #                               image_size=temp_image_size,
        #                               batch_size=temp_batch_size,
        #                               num_workers=temp_num_workers,
        #                               mode='test',
        #                               augmentation_prob=0.)
        with torch.no_grad(): # 测试时候不需要梯度回传 节省空间
            self.unet.train(False)
            self.unet.eval()


            for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):
                # degth_cm = landmark_depth[num - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                image = images
                # images = self.img_catdist_channel(images)
                images = images.to(self.device)
                GT = GT.to(self.device, torch.long)
                # SR = F.sigmoid(self.unet(images))
                # t1 = time.time()
                # SR_lm = self.unet(images)
                # SR = F.sigmoid(SR)
                # SR2 = F.sigmoid(SR_r)
                # SR = torch.cat((SR_l, SR_l, SR_r), dim=1)
                # GT_sg = self.onehot_to_mulchannel(GT)
                GT = GT.squeeze(1)
                # print('max:', GT.max())

                GT_0 = GT == 0  # 0代表其余部分
                GT_1 = GT == 76  # 1代表耻骨联合
                GT_2 = GT == 150  # 2代表胎头

                # GT2 = GT2.squeeze(1)
                # GT2_0 = GT2 == 0  # 0代表其余部分
                # GT2_1 = GT2 == 1  # 1代表耻骨联合
                # GT2_2 = GT2 == 2  # 2代表胎头
                #
                # GT3 = GT3.squeeze(1)
                # GT3_0 = GT3 == 0  # 0代表其余部分
                # GT3_1 = GT3 == 1  # 1代表耻骨联合
                # GT3_2 = GT3 == 2  # 2代表胎头
                #
                # GT4 = GT4.squeeze(1)
                # GT4_0 = GT4 == 0  # 0代表其余部分
                # GT4_1 = GT4 == 1  # 1代表耻骨联合
                # GT4_2 = GT4 == 2  # 2代表胎头
                #
                # GT5 = GT5.squeeze(1)
                # GT5_0 = GT5 == 0  # 0代表其余部分
                # GT5_1 = GT5 == 1  # 1代表耻骨联合
                # GT5_2 = GT5 == 2  # 2代表胎头

                GT = torch.cat((GT_0.unsqueeze(1), GT_1.unsqueeze(1), GT_2.unsqueeze(1)), 1).int()
                # print(GT.max())
                print(cor_num)
                # yu的图片要用012 新数据用sun
                # GT = self.onehot_to_mulchannel_sun(GT)
                # GT = self.onehot_to_mulchannel(GT)


                # GT2 = torch.cat((GT2_0.unsqueeze(1), GT2_1.unsqueeze(1), GT2_2.unsqueeze(1)), 1).int()
                # GT3 = torch.cat((GT3_0.unsqueeze(1), GT3_1.unsqueeze(1), GT3_2.unsqueeze(1)), 1).int()
                # GT4 = torch.cat((GT4_0.unsqueeze(1), GT4_1.unsqueeze(1), GT4_2.unsqueeze(1)), 1).int()
                # GT5 = torch.cat((GT5_0.unsqueeze(1), GT5_1.unsqueeze(1), GT5_2.unsqueeze(1)), 1).int()

                # SR_lm, SR_seg, _, _, _, _ = self.unet(images)
                # SR_seg = self.unet(images)
                # SR_seg, SR_seg2, SR_seg3, SR_seg4, SR_seg5 = self.unet(images)
                if False:
                    SR_seg, SR_seg5 = self.unet(images)
                # SR_seg, SR_seg5 = self.unet(images)
                SR_seg, g1, g2, g3 = self.unet(images)
                # SR_seg = self.unet(images)
                # SR_seg=SR_lm
                # SR_lm = torch.sigmoid(SR_lm)
                SR_seg = torch.softmax(SR_seg, 1) > 0.5
                # g1 = torch.softmax(g1, 1) > 0.5
                # g2 = torch.softmax(g2, 1) > 0.5
                # g3 = torch.softmax(g3, 1) > 0.5

                SR_seg = SR_seg.int()

                Dst1 = g1.mul(255)
                Dst2 = g2.mul(255)
                Dst3 = g3.mul(255)

                Dst1 = Dst1.int()
                Dst2 = Dst2.int()
                Dst3 = Dst3.int()



                Src = GT.mul(255)
                Dst = SR_seg.mul(255)
                


                # contours, _ = cv.findContours(SR[:, :, 1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

                # temp_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/'.format(dataset_no))
                # temp_path = os.path.join(self.model_path, 'final_results/dataset_test/unet_image/'.format(dataset_no))
                # temp_path = os.path.join(self.model_path, 'final_results/{}/unet_image/'.format(self.model_type))
                temp_path = 'paper/g123' #为每个模型进行测试并保存
                if  os.path.exists(temp_path) is False:
                    os.makedirs(temp_path)
                    print('succeed to mkdirs: {}'.format(temp_path))

                for index in range(0, len(filename)):
                    # for index in range(0, len(cor_num)):
                    # filename = filename[0]
                    # for i in range(0, 3): # src是GT
                    #     # np.expand_dims(0)
                    #     image_item = Src[index][i].cpu().detach().numpy().astype(np.uint8)
                    #     image_item = Image.fromarray(image_item, mode='L')
                    #     # image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i)))
                    #     image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i)))
                    #     print('-------------------std-to-real-----------------------')
                        # image_item.save('./result/pictures_DBU_1_1_1_single_0.2_upsamle_defnorm_conv/ATD-{}-src-{}.png'.format(cor_num[index], i))
                    #predict
                    image_item = Dst[index][0].cpu().detach().numpy().astype(np.uint8)
                    image_item = Image.fromarray(image_item, mode='L')
                    # image_item.save(os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(cor_num[index], i)))
                    image_item.save(os.path.join(temp_path, 'ATD-{}-predict.png'.format(filename[index])))

                    #g1
                    image_item_g1 = Dst1[index][0].cpu().detach().numpy().astype(np.uint8)
                    image_item_g1 = Image.fromarray(image_item_g1, mode='L')
                    # image_item.save(os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(cor_num[index], i)))
                    image_item_g1.save(os.path.join(temp_path, 'ATD-{}-g1.png'.format(filename[index])))

                    #g2
                    image_item_g2 = Dst2[index][0].cpu().detach().numpy().astype(np.uint8)
                    image_item_g2 = Image.fromarray(image_item_g2, mode='L')
                    # image_item.save(os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(cor_num[index], i)))
                    image_item_g2.save(os.path.join(temp_path, 'ATD-{}-g2.png'.format(filename[index])))

                    #g3
                    image_item_g3 = Dst3[index][0].cpu().detach().numpy().astype(np.uint8)
                    image_item_g3 = Image.fromarray(image_item_g3, mode='L')
                    # image_item.save(os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(cor_num[index], i)))
                    image_item_g3.save(os.path.join(temp_path, 'ATD-{}-g3.png'.format(filename[index])))
                    break


                print("generate batch {}".format(len(cor_num)))
                # print(filename)
                break


            print("generate end ......")

    def test_model_py(self):
        #加载模型直接测试
        # unet_path = './result/model_weight_4000/Multi_U_Net_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/My_Unet_no_multi_has_shape_no_attention_0.0001_1.pth'
        unet_path = './result/model_weight_4000/My_Unet_has_multi_has_shape_no_attention_0.0001_1.pth'
        # unet_path = './result/model_weight_4000/AG_DBU_YS_Net_16_0.0001_1.pth'
        unet_path = './result/model_weight_4000/{}_0.0001_1.pth'.format(self.model_type)

        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
            self.unet.to(self.device)
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')
            return

        with torch.no_grad(): # 测试时候不需要梯度回传 节省空间
            self.unet.train(False)
            self.unet.eval()

            temp_path = 'paper/test_speed/{}'.format(self.model_type) #为每个模型进行测试并保存
            if  os.path.exists(temp_path) is False:
                os.makedirs(temp_path)
                print('succeed to mkdirs: {}'.format(temp_path))
            time_all = 0
            for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):
                time1 = time.perf_counter()
                # time1 = time.clock()
                # time1 = time.process_time()
                # time1 = time.time()
                # degth_cm = landmark_depth[num - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                # image = images
                # images = self.img_catdist_channel(images)
                images = images.to(self.device)
                #GT = GT.to(self.device, torch.long)
                # SR = F.sigmoid(self.unet(images))
                # t1 = time.time()
                # SR_lm = self.unet(images)
                # SR = F.sigmoid(SR)
                # SR2 = F.sigmoid(SR_r)
                # SR = torch.cat((SR_l, SR_l, SR_r), dim=1)
                # GT_sg = self.onehot_to_mulchannel(GT)
                #GT = GT.squeeze(1)
                # print('max:', GT.max())

                # GT_0 = GT == 0  # 0代表其余部分
                # GT_1 = GT == 76  # 1代表耻骨联合
                # GT_2 = GT == 150  # 2代表胎头
                # GT = torch.cat((GT_0.unsqueeze(1), GT_1.unsqueeze(1), GT_2.unsqueeze(1)), 1).int()
                # print(GT.max())


                # SR_lm, SR_seg, _, _, _, _ = self.unet(images)
                # SR_seg = self.unet(images)
                # SR_seg, SR_seg2, SR_seg3, SR_seg4, SR_seg5 = self.unet(images)
                # SR_seg, SR_seg5 = self.unet(images)

                #my unet
                # SR_seg, _, _, _ = self.unet(images)

                # 单个返回值 包括sau
                SR_seg = self.unet(images)

                #MDunet
                # SR_lm, SR_seg, SR_lm_d2, SR_lm_d4, SR_lm_d8, logsigma = self.unet(images)

                SR_seg = torch.softmax(SR_seg, 1) > 0.5


                SR_seg = SR_seg.int()





                # Src = GT.mul(255)
                Dst = SR_seg.mul(255)


                for index in range(0, len(filename)):
                    # for index in range(0, len(cor_num)):
                    # filename = filename[0]
                    # for i in range(0, 3): # src是GT
                    #     # np.expand_dims(0)
                    #     image_item = Src[index][i].cpu().detach().numpy().astype(np.uint8)
                    #     image_item = Image.fromarray(image_item, mode='L')
                    #     # image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i)))
                    #     image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(filename[index], i)))
                    #     print('-------------------std-to-real-----------------------')
                    # image_item.save('./result/pictures_DBU_1_1_1_single_0.2_upsamle_defnorm_conv/ATD-{}-src-{}.png'.format(cor_num[index], i))
                    #predict
                    image_item = Dst[index][0].cpu().detach().numpy().astype(np.uint8)
                    image_item = Image.fromarray(image_item, mode='L')
                    # image_item.save(os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(cor_num[index], i)))
                    image_item.save(os.path.join(temp_path, 'ATD-{}-predict.png'.format(filename[index])))


                    break

                # time2 = time.clock()
                # time2 = time.process_time()
                # time2 = time.time()
                time2 = time.perf_counter()
                time_all = time_all + time2 - time1
                # print("generate batch {}".format(len(cor_num)))
                # print(filename)

            # time_end = time.clock()
            # time_seg = time_end - time_begin
            print('seg time: {}s'.format(time_all))
            print("generate end ......")

    def test_aop_speed(self):


        self.image_size_h = 384


        temp_path = os.path.join('aop_results/{}/unet_image/'.format(self.model_type)) #为每个模型进行测试并保存
        aop_result_path = temp_path.split('unet_image')[0]
        if  os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
            print('succeed to mkdirs: {}'.format(temp_path))
        time_begin = time.clock()
        for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):

            for index in range(0, len(filename)):
                left_point = right_point = tangent_point = []


                temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], 0))
                # temp_path_item = os.path.join(temp_path, '623-{}-GT-{}.png'.format(filename[0], 0))
                img_background = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)
                img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGR)



                # 获得预测图片的长轴端点和切点
                flag = 0 #图片出错 如全黑 就跳过这个预测
                for i in range(1, 3):
                    # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                    temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i))
                    # print(temp_path_item)
                    img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                    _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    max_contour = self.get_max_contour(contours)
                    # print(max_contour)
                    if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                        print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                        flag = 1
                        break
                    ellipse_instance = cv2.fitEllipse(max_contour[0])
                    cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 1)
                    if i == 1:
                        left_point, right_point = get_endpoint(ellipse_instance)
                    else:
                        tangent_point = get_tangent_point(right_point, ellipse_instance)

                if flag == 1:
                    flag == 0
                    continue

                # 计算距离并保存
                # distance = pixel_mm * math.sqrt(math.pow(right_point[0] - left_point[0], 2) + math.pow(right_point[1] - left_point[1], 2))
                cv2.line(img_background, (round(left_point[0]), round(left_point[1])),
                         (round(right_point[0]), round(right_point[1])), (255, 255, 0), 1)
                # cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 2)
                cv2.line(img_background, (round(right_point[0]), round(right_point[1])),
                         (round(tangent_point[0]), round(tangent_point[1])), (0, 0, 255), 1)
                temp_path_item = os.path.join(temp_path, 'ATD-{}-real_marked_result-{}.png'.format(filename[index], 0))
                cv2.imwrite(temp_path_item, img_background)

                pic_no = 'ATD-marked_result-{}'.format(filename[index])



                # 计算预测的aop角度 角度计算 a · b = |a| * |b| * cos(o)

                vector_output_axis = (left_point[0] - right_point[0], left_point[1] - right_point[1])

                vector_output_tangent = (tangent_point[0] - right_point[0], tangent_point[1] - right_point[1])

                top = vector_output_axis[0] * vector_output_tangent[0] + vector_output_axis[1] * vector_output_tangent[
                    1]
                bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
                bottom2 = math.sqrt(math.pow(vector_output_tangent[0], 2) + math.pow(vector_output_tangent[1], 2))
                output_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)
        time_end = time.clock()
        time_all = time_end - time_begin
        print('aop time: {}s'.format(time_all))
        print("generate end ......")


    #othrers
    def generate_result_75(self):
        # result_history = pd.DataFrame(
        #     columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_pixel", "right_point_distance", "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff"])
        # for dataset_no in range(1, 6):
        #     unet_path = os.path.join(self.model_path,
        #                              'final_models/v_net/dataset{}/DBU_YS_Net_16.pkl'.format(dataset_no))
        #     ####
        #     self.build_model()
        #     if os.path.isfile(unet_path):
        #         # Load the pretrained Encoder
        #         self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
        #         print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        #     else:
        #         print('No pretrained_model')
        #     self.generate_output_pic(dataset_no)
        #     self.generate_output_aop(dataset_no)
        result_history = pd.DataFrame(
            columns=["dataset_no", "left_pixel_mean", "left_pixel_median", "left_pixel_var", "left_distance_mean", "left_distance_median", "left_distance_var",
                     "right_pixel_mean", "right_pixel_median", "right_pixel_var", "right_distance_mean", "right_distance_median", "right_distance_var",
                     "axis_diff_mean", "axis_diff_median", "axis_diff_var", "aop_diff_mean", "aop_diff_median", "aop_diff_var"]
        )
        count = 0
        for dataset_no in range(1, 2):# final_results/new_0.715_to_0.75/AG_DBU_old/dataset{}/
            # csv_path = os.path.join(self.model_path, 'final_results/dataset{}/aop_history.csv'.format(dataset_no))
            csv_path = os.path.join(self.model_path, 'final_results/dataset{}/unet_image/aop_history.csv'.format(dataset_no))
            item_history = pd.read_csv(csv_path)
            list_pixel = item_history['left_point_pixel']
            left_pixel_mean, left_pixel_median, left_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['left_point_distance']
            left_distance_mean, left_distance_median, left_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_pixel']
            right_pixel_mean, right_pixel_median, right_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_distance']
            right_distance_mean, right_distance_median, right_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['axis_angle_diff']
            axis_diff_mean, axis_diff_median, axis_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['aop_diff']
            aop_diff_mean, aop_diff_median, aop_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))

            info = (dataset_no, left_pixel_mean, left_pixel_median, left_pixel_var, left_distance_mean, left_distance_median, left_distance_var,
                    right_pixel_mean, right_pixel_median, right_pixel_var, right_distance_mean, right_distance_median, right_distance_var,
                    axis_diff_mean, axis_diff_median, axis_diff_var, aop_diff_mean, aop_diff_median, aop_diff_var)
            result_history.loc[count] = info
            count = count + 1

        temp_path = os.path.join(self.model_path, 'final_results/')
        result_history.to_csv(temp_path + 'result_history.csv', index=False)
        print("genenrate result_history success ......")

    def generate_result(self):
        # result_history = pd.DataFrame(
        #     columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_pixel", "right_point_distance", "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff"])
        for dataset_no in range(1, 6):
            unet_path = os.path.join(self.model_path,
                                     'final_models/BiSeNetV2/dataset{}/BiSeNetV2.pkl'.format(dataset_no))
            ####
            self.build_model()
            if os.path.isfile(unet_path):
                # Load the pretrained Encoder
                self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
                print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
            else:
                print('No pretrained_model')
            self.generate_output_pic(dataset_no)
            self.generate_output_aop_75(dataset_no)

        result_history = pd.DataFrame(
            columns=["dataset_no", "left_pixel_mean", "left_pixel_median", "left_pixel_var", "left_distance_mean", "left_distance_median", "left_distance_var",
                     "right_pixel_mean", "right_pixel_median", "right_pixel_var", "right_distance_mean", "right_distance_median", "right_distance_var",
                     "axis_diff_mean", "axis_diff_median", "axis_diff_var", "aop_diff_mean", "aop_diff_median", "aop_diff_var", "aop_diff_std"]
        )
        count = 0
        for dataset_no in range(1, 6):
            csv_path = os.path.join(self.model_path, 'final_results/new_0.715_to_0.75/BiSeNetV2/dataset{}/aop_history.csv'.format(dataset_no))
            item_history = pd.read_csv(csv_path)
            list_pixel = item_history['left_point_pixel']
            left_pixel_mean, left_pixel_median, left_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['left_point_distance']
            left_distance_mean, left_distance_median, left_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_pixel']
            right_pixel_mean, right_pixel_median, right_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_distance']
            right_distance_mean, right_distance_median, right_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['axis_angle_diff']
            axis_diff_mean, axis_diff_median, axis_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['aop_diff']
            aop_diff_mean, aop_diff_median, aop_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            aop_diff_std = math.sqrt(aop_diff_var)
            info = (dataset_no, left_pixel_mean, left_pixel_median, left_pixel_var, left_distance_mean, left_distance_median, left_distance_var,
                    right_pixel_mean, right_pixel_median, right_pixel_var, right_distance_mean, right_distance_median, right_distance_var,
                    axis_diff_mean, axis_diff_median, axis_diff_var, aop_diff_mean, aop_diff_median, aop_diff_var, aop_diff_std)
            # if dataset_no == 1:
            #     temp_info = info
            # else:
            #     temp_info = temp_info + info
            result_history.loc[count] = info
            count = count + 1
        result_history.loc[count] = (result_history.loc[0] + result_history.loc[1] + result_history.loc[2] + result_history.loc[3] + result_history.loc[4])/5
        result_history.loc[count][0] = 100
        temp_path = os.path.join(self.model_path, 'final_results/new_0.715_to_0.75/BiSeNetV2/')
        result_history.to_csv(temp_path + 'result_history.csv', index=False)
        print("genenrate result_history success ......")


    def generate_paper_pic(self):
        # temp_models\final_models\AG_DBU_old\dataset1
        dataset_no = 1
        unet_path = os.path.join(self.model_path,
                                 'final_models/AG_DBU_old/dataset{}/DBU_YS_Net_16.pkl'.format(dataset_no))
        ####
        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')
        temp_test_path = './dataset_{}/valid/valid'.format(dataset_no)
        temp_image_size = 512
        temp_batch_size = 1
        temp_num_workers = 1
        self.test_loader = get_loader(image_path=temp_test_path,
                                      image_size=temp_image_size,
                                      batch_size=temp_batch_size,
                                      num_workers=temp_num_workers,
                                      mode='test',
                                      augmentation_prob=0.)
        self.unet.train(False)
        self.unet.eval()
        # 读取深度文件
        # depth_Path = 'D:/py_seg/Landmark-Net/depth.csv'
        # landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        pixel_num = self.image_size * 0.715
        a_5 = 0
        a_10 = 0
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        list_aop = []
        list_l = []
        list_r = []
        list_aop_root_mse = []
        out_aop = []
        true_aop = []
        for i, (images, GT, GT5, GT_Lmark, cor_num) in enumerate(self.test_loader):
            # degth_cm = landmark_depth[num - 1]
            # pixel_mm = degth_cm * 10 / pixel_num
            image = images
            # images = self.img_catdist_channel(images)
            images = images.to(self.device)
            GT = GT.to(self.device, torch.long)
            # SR = F.sigmoid(self.unet(images))
            # t1 = time.time()
            # SR_lm = self.unet(images)
            # SR = F.sigmoid(SR)
            # SR2 = F.sigmoid(SR_r)
            # SR = torch.cat((SR_l, SR_l, SR_r), dim=1)
            # GT_sg = self.onehot_to_mulchannel(GT)
            GT = GT.squeeze(1)
            GT_0 = GT == 0  # 0代表其余部分
            GT_1 = GT == 1  # 1代表耻骨联合
            GT_2 = GT == 2  # 2代表胎头

            # GT2 = GT2.squeeze(1)
            # GT2_0 = GT2 == 0  # 0代表其余部分
            # GT2_1 = GT2 == 1  # 1代表耻骨联合
            # GT2_2 = GT2 == 2  # 2代表胎头
            #
            # GT3 = GT3.squeeze(1)
            # GT3_0 = GT3 == 0  # 0代表其余部分
            # GT3_1 = GT3 == 1  # 1代表耻骨联合
            # GT3_2 = GT3 == 2  # 2代表胎头
            #
            # GT4 = GT4.squeeze(1)
            # GT4_0 = GT4 == 0  # 0代表其余部分
            # GT4_1 = GT4 == 1  # 1代表耻骨联合
            # GT4_2 = GT4 == 2  # 2代表胎头
            #
            # GT5 = GT5.squeeze(1)
            # GT5_0 = GT5 == 0  # 0代表其余部分
            # GT5_1 = GT5 == 1  # 1代表耻骨联合
            # GT5_2 = GT5 == 2  # 2代表胎头

            GT = torch.cat((GT_0.unsqueeze(1), GT_1.unsqueeze(1), GT_2.unsqueeze(1)), 1).int()

            # GT2 = torch.cat((GT2_0.unsqueeze(1), GT2_1.unsqueeze(1), GT2_2.unsqueeze(1)), 1).int()
            # GT3 = torch.cat((GT3_0.unsqueeze(1), GT3_1.unsqueeze(1), GT3_2.unsqueeze(1)), 1).int()
            # GT4 = torch.cat((GT4_0.unsqueeze(1), GT4_1.unsqueeze(1), GT4_2.unsqueeze(1)), 1).int()
            # GT5 = torch.cat((GT5_0.unsqueeze(1), GT5_1.unsqueeze(1), GT5_2.unsqueeze(1)), 1).int()

            # SR_lm, SR_seg, _, _, _, _ = self.unet(images)
            # SR_seg = self.unet(images)
            # SR_seg, SR_seg2, SR_seg3, SR_seg4, SR_seg5 = self.unet(images)
            if True:
                # return seg1, seg5, x1, self.temp_2_1, self.temp_temp_lower, self.temp_temp_ag
                # SR_seg, SR_seg5 = self.unet(images)
                SR_seg, SR_seg5, x1, temp_upper, temp_temp_lower, temp_temp_ag = self.unet(images)

            # SR_seg=SR_lm
            # SR_lm = torch.sigmoid(SR_lm)
            SR_seg = torch.softmax(SR_seg, 1) > 0.5
            # SR_seg2 = torch.softmax(SR_seg2, 1) > 0.5
            # SR_seg3 = torch.softmax(SR_seg3, 1) > 0.5
            # SR_seg4 = torch.softmax(SR_seg4, 1) > 0.5
            if False:
                SR_seg5 = torch.softmax(SR_seg5, 1) > 0.5

            SR_seg = SR_seg.int()
            # SR_seg2 = SR_seg2.int()
            # SR_seg3 = SR_seg3.int()
            # SR_seg4 = SR_seg4.int()
            if False:
                SR_seg5 = SR_seg5.int()

            Src = GT.mul(255)
            Dst = SR_seg.mul(255)
            # Dst2 = SR_seg2.mul(255)
            # Dst3 = SR_seg3.mul(255)
            # Dst4 = SR_seg4.mul(255)
            if False:
                Dst5 = SR_seg5.mul(255)
            # contours, _ = cv.findContours(SR[:, :, 1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # temp_models\paper_ag\dataset1
            temp_path = os.path.join(self.model_path, 'paper_ag/dataset{}/'.format(dataset_no))
            x1 = x1.mul(255)
            x1 = x1.int()
            # temp_temp_ag = temp_temp_ag[0]
            x1 = x1[0].cpu().detach().numpy().astype(np.uint8)
            for index in range(0, len(x1)):
                image_item = x1[index]
                image_item = Image.fromarray(image_item, mode='L')
                image_item.save(os.path.join(temp_path, 'ATD{}--{}-x1.png'.format(cor_num[i], index)))

            temp_upper = temp_upper.mul(255)
            temp_upper = temp_upper.int()
            # temp_temp_ag = temp_temp_ag[0]
            temp_upper = temp_upper[0].cpu().detach().numpy().astype(np.uint8)
            for index in range(0, len(temp_upper)):
                image_item = temp_upper[index]
                image_item = Image.fromarray(image_item, mode='L')
                image_item.save(os.path.join(temp_path, 'ATD{}--{}-upper.png'.format(cor_num[i], index)))

            temp_temp_lower = temp_temp_lower.mul(255)
            temp_temp_lower = temp_temp_lower.int()
            # temp_temp_ag = temp_temp_ag[0]
            temp_temp_lower = temp_temp_lower[0].cpu().detach().numpy().astype(np.uint8)
            for index in range(0, len(temp_temp_lower)):
                image_item = temp_temp_lower[index]
                image_item = Image.fromarray(image_item, mode='L')
                image_item.save(os.path.join(temp_path, 'ATD{}--{}-lower.png'.format(cor_num[i], index)))

            temp_temp_ag = temp_temp_ag.mul(255)
            temp_temp_ag = temp_temp_ag.int()
            # temp_temp_ag = temp_temp_ag[0]
            temp_temp_ag = temp_temp_ag[0].cpu().detach().numpy().astype(np.uint8)
            for index in range(0, len(temp_temp_ag)):
                image_item = temp_temp_ag[index]
                image_item = Image.fromarray(image_item, mode='L')
                image_item.save(os.path.join(temp_path, 'ATD{}--{}-ag.png'.format(cor_num[i], index)))
                print("generate:{}".format(index))
            # image_item = temp_temp_ag[0][0].cpu().detach().numpy().astype(np.uint8)
            # image_item = Image.fromarray(image_item, mode='L')
            # image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(0, 0)))
            # for index in range(0, len(cor_num)):
            #     for i in range(0, 3):
            #         # np.expand_dims(0)
            #         image_item = Src[index][i].cpu().detach().numpy().astype(np.uint8)
            #         image_item = Image.fromarray(image_item, mode='L')
            #         image_item.save(os.path.join(temp_path, 'ATD-{}-standard-{}.png'.format(cor_num[index], i)))
            #         # image_item.save('./result/pictures_DBU_1_1_1_single_0.2_upsamle_defnorm_conv/ATD-{}-src-{}.png'.format(cor_num[index], i))
            #     for i in range(0, 3):
            #         image_item = Dst[index][i].cpu().detach().numpy().astype(np.uint8)
            #         image_item = Image.fromarray(image_item, mode='L')
            #         image_item.save(os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i)))
            # print("generate batch {}".format(len(cor_num)))
            break
        print("generate end ......")
        return 0
        # result_history = pd.DataFrame(
        #     columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_pixel", "right_point_distance", "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff"])
        for dataset_no in range(1, 6):
            unet_path = os.path.join(self.model_path,
                                     'final_models/DBU/dataset{}/DBU_YS_Net_16.pkl'.format(dataset_no))
            ####
            self.build_model()
            if os.path.isfile(unet_path):
                # Load the pretrained Encoder
                self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
                print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
            else:
                print('No pretrained_model')
            self.generate_output_pic(dataset_no)
            self.generate_output_aop(dataset_no)

        result_history = pd.DataFrame(
            columns=["dataset_no", "left_pixel_mean", "left_pixel_median", "left_pixel_var", "left_distance_mean", "left_distance_median", "left_distance_var",
                     "right_pixel_mean", "right_pixel_median", "right_pixel_var", "right_distance_mean", "right_distance_median", "right_distance_var",
                     "axis_diff_mean", "axis_diff_median", "axis_diff_var", "aop_diff_mean", "aop_diff_median", "aop_diff_var"]
        )
        count = 0
        for dataset_no in range(1, 6):
            csv_path = os.path.join(self.model_path, 'final_results/dataset{}/aop_history.csv'.format(dataset_no))
            item_history = pd.read_csv(csv_path)
            list_pixel = item_history['left_point_pixel']
            left_pixel_mean, left_pixel_median, left_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['left_point_distance']
            left_distance_mean, left_distance_median, left_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_pixel']
            right_pixel_mean, right_pixel_median, right_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_distance']
            right_distance_mean, right_distance_median, right_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['axis_angle_diff']
            axis_diff_mean, axis_diff_median, axis_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['aop_diff']
            aop_diff_mean, aop_diff_median, aop_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))

            info = (dataset_no, left_pixel_mean, left_pixel_median, left_pixel_var, left_distance_mean, left_distance_median, left_distance_var,
                    right_pixel_mean, right_pixel_median, right_pixel_var, right_distance_mean, right_distance_median, right_distance_var,
                    axis_diff_mean, axis_diff_median, axis_diff_var, aop_diff_mean, aop_diff_median, aop_diff_var)
            result_history.loc[count] = info
            count = count + 1

        temp_path = os.path.join(self.model_path, 'final_results/')
        result_history.to_csv(temp_path + 'result_history.csv', index=False)
        print("genenrate result_history success ......")


    def generate_result_dataset_no(self, dataset_no_flag=0):
        if dataset_no_flag == 0:
            print("lack of dataset_no")
            return
        # result_history = pd.DataFrame(
        #     columns=["pic_no", "left_point_pixel", "left_point_distance", "right_point_pixel", "right_point_pixel", "right_point_distance", "tangent_point_pixel", "tangent_point_distance", "axis_angle_diff", "aop_diff"])
        for dataset_no in range(dataset_no_flag, dataset_no_flag+1):
            unet_path = os.path.join(self.model_path,
                                     'final_models/dataset{}/DBU_YS_Net_16.pkl'.format(dataset_no))
            ####
            self.build_model()
            if os.path.isfile(unet_path):
                # Load the pretrained Encoder
                self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))
                print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
            else:
                print('No pretrained_model')
            self.generate_output_pic(dataset_no)
            self.generate_output_aop(dataset_no)

        result_history = pd.DataFrame(
            columns=["dataset_no", "left_pixel_mean", "left_pixel_median", "left_pixel_var", "left_distance_mean", "left_distance_median", "left_distance_var",
                     "right_pixel_mean", "right_pixel_median", "right_pixel_var", "right_distance_mean", "right_distance_median", "right_distance_var",
                     "axis_diff_mean", "axis_diff_median", "axis_diff_var", "aop_diff_mean", "aop_diff_median", "aop_diff_var"]
        )
        count = 0
        for dataset_no in range(1, 6):
            csv_path = os.path.join(self.model_path, 'final_results/dataset{}/aop_history.csv'.format(dataset_no))
            item_history = pd.read_csv(csv_path)
            list_pixel = item_history['left_point_pixel']
            left_pixel_mean, left_pixel_median, left_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['left_point_distance']
            left_distance_mean, left_distance_median, left_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_pixel']
            right_pixel_mean, right_pixel_median, right_pixel_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['right_point_distance']
            right_distance_mean, right_distance_median, right_distance_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['axis_angle_diff']
            axis_diff_mean, axis_diff_median, axis_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))
            list_pixel = item_history['aop_diff']
            aop_diff_mean, aop_diff_median, aop_diff_var = (np.mean(list_pixel), np.median(list_pixel), np.var(list_pixel))

            info = (dataset_no, left_pixel_mean, left_pixel_median, left_pixel_var, left_distance_mean, left_distance_median, left_distance_var,
                    right_pixel_mean, right_pixel_median, right_pixel_var, right_distance_mean, right_distance_median, right_distance_var,
                    axis_diff_mean, axis_diff_median, axis_diff_var, aop_diff_mean, aop_diff_median, aop_diff_var)
            result_history.loc[count] = info
            count = count + 1

        temp_path = os.path.join(self.model_path, 'final_results/')
        result_history.to_csv(temp_path + 'result_history.csv', index=False)
        print("genenrate result_history success ......")

    def test_output_pic_hc(self):
        unet_path = os.path.join(self.model_path, '128-fetalhead-bisenet-BiSeNet-199-0.0001-139-0.5000-128.pkl')
        save_path = r'D:\py_seg\Landmark-Net\result\pic_output\fetalhead_result\128bisenet/'
        ####
        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        self.unet.train(False)
        self.unet.eval()

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        DC_1 = 0.
        DC_2 = 0.
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        # list_aop = []
        for i, (images, GT, img_name) in enumerate(self.valid_loader):
            img_name = img_name[0]
            print(img_name)

            image = images
            print(images.shape)
            # images = self.img_catdist_channel(images)
            images = images.to(self.device)
            GT = GT.to(self.device, torch.long)
            # SR = F.sigmoid(self.unet(images))
            # t1 = time.time()
            SR_seg = self.unet(images)
            SR_seg = torch.sigmoid(SR_seg)
            # SR2 = F.sigmoid(SR_r)
            # SR = torch.cat((SR_l, SR_l, SR_r), dim=1)
            # GT_sg = self.onehot_to_mulchannel(GT)

            # SR_seg=SR_lm
            # SR_lm = torch.sigmoid(SR_lm)
            # SR_seg = torch.softmax(SR_seg, 1)

            acc += get_accuracy(SR_seg, GT)
            SE += get_sensitivity(SR_seg, GT)
            SP += get_specificity(SR_seg, GT)
            PC += get_precision(SR_seg, GT)
            F1 += get_F1(SR_seg, GT)
            JS += get_JS(SR_seg, GT)
            DC += get_DC(SR_seg, GT)
            length += 1

            SR_seg = SR_seg > 0.5
            # SR_sp = SR[:, 1, :,:].mul(255)
            # SR_sp = SR_sp.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # sp_j = np.zeros((SR_sp.shape[0], SR_sp.shape[1],3)).astype(np.uint8)
            # sp_j[:,:,2] = SR_sp
            # SR_h = SR[:, 2, :, :].mul(255)
            # SR_h = SR_h.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # hd_j = np.zeros((SR_h.shape[0], SR_h.shape[1],3)).astype(np.uint8)
            # hd_j[:,:,2] = SR_h
            # sp_j[:, :, 1] = SR_h
            # GT_sg1 = GT_sg[:, 1, :,:].mul(255)
            # GT_sg1 = GT_sg1.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # GT_sg2 = GT_sg[:, 2, :, :].mul(255)
            # GT_sg2 = GT_sg2.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # GT_j = np.zeros((GT_sg1.shape[0], GT_sg1.shape[1], 3)).astype(np.uint8)
            # GT_j[:, :, 2] = GT_sg1
            # GT_j[:, :, 1] = GT_sg2
            SR_seg = SR_seg.mul(255)
            SR_seg = SR_seg.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
            # lm_j = np.zeros((lm_h.shape[0], lm_h.shape[1], 3)).astype(np.uint8)
            # SR_seg = SR_seg[:,:,[2,1,0]]
            GT = GT.mul(255)
            GT = GT.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
            image = (image.mul(127)) + 128
            image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
        # image1 = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        # image_h = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        ########landmark
        # SR1 = SR_lm[0, 0, :, :].cpu().detach()
        # SR2 = SR_lm[0, 1, :, :].cpu().detach()
        # GT1 = GT_Lmark[0, 0, :, :].cpu().detach()
        # GT2 = GT_Lmark[0, 1, :, :].cpu().detach()
        # GT3 = GT_Lmark[0, 2, :, :].cpu().detach()

        #
        # print('[aop] aopd: %.4f, median: %.4f, mean: %.4f,std: %.4f' % (abs(Aod-aop),np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
        # print('[std] left: %.4f, right: %.4f,Angle: %.4f' % (np.std(list_l), np.std(list_r), np.std(list_angle)))
        # cv.imwrite(r'D:\py_seg\Landmark-Net\result\exp_output\multiwoafm\seg/' + str(num).zfill(4) + 'seg.png', sp_j)
        # cv.imwrite(r'D:\py_seg\Landmark-Net\result\exp_output\single\locjet/' + str(num).zfill(4) + 'lmjet.png', img_lm)

        # cv.imwrite(r'D:/py_seg/U-Net/U-Net_vari/result/pic_output/landmark2/' + str(i) + '_r.png', SR_h)
        # cv.imwrite(save_path + img_name + '_img.png', image)
        # cv.imwrite(save_path + img_name + '_gt.png', GT)
        # cv.imwrite(save_path + img_name + '_sr.png', SR_seg)
        # cv.imwrite(r'D:\py_seg\Landmark-Net\result\exp_output\single\aop/' + str(num).zfill(4) + 'mulaop.png', img_result)
        # cv.imwrite(r'D:/py_seg/U-Net/U-Net_vari/result/pic_output/' + str(i) + '_result_noaop.png', img_result)
        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length

        print(
            '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC, DC, DC))
        # I=cv.imread(r'D:\py_seg\U-Net\U-Net_vari\dataset\test\ATD_0004.png')
        # print('[aopall] median: %.4f, mean: %.4f,std: %.4f' % (np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
        # print('[ag-num] total image: %d  ag_5: %d, ag_10: %d' % (length, a_5, a_10))
        net_input = torch.rand(1, 1, images.shape[2], images.shape[3]).to(self.device)
        print(net_input.size())
        flops, params = profile(self.unet, inputs=(net_input,))
        print("flops: %.f  parmas: %.f", (flops / 1000000000, params / (1024 * 1024)))

    def saveONNX(self, filepath, model_name):
        '''
        保存ONNX模型
        :param model: 神经网络模型
        :param filepath: 文件保存路径
        '''
        unet_path = os.path.join(self.model_path, model_name)
        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        self.unet.train(False)
        self.unet.eval()
        # 神经网络输入数据类型
        dummy_input = torch.randn(1, 1, 320, 512, device='cuda')
        torch.onnx.export(self.unet, dummy_input, filepath, verbose=True)

    def edgefliter(self, approxCurve, turn_angle=60):
        maxindex1 = 0
        max1 = 0
        angle_list = []
        angle_sign = []
        seg_index = [0, ]
        ellip_group = []
        seged_edgeset = []
        # img_med = cv.medianBlur(img, 7)	# 滤波核7，9

        # contours, _ = cv.findContours(img_med, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        for i in range(1, len(approxCurve) - 1):
            lineseg1 = (np.array(approxCurve[i]) - np.array(approxCurve[i - 1])).squeeze(0)
            lineseg2 = (np.array(approxCurve[i + 1]) - np.array(approxCurve[i])).squeeze(0)
            L_1 = np.sqrt(lineseg1.dot(lineseg1))
            L_2 = np.sqrt(lineseg2.dot(lineseg2))

            pi_dist = lineseg1.dot(lineseg2) / (L_1 * L_2)
            if abs(pi_dist) >= 1.:
                # pi_dist = (pi_dist / abs(pi_dist)) * 0.999  # 防止arccos输入超出范围[-1,1]  判断是否超出1
                print('pi_dist value overflow')
            angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi
            angle_list.append(angle_dist)
            a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])  # 不用计算出角度再判断正负，直接用叉乘值判断正负？
            angle_sign.append(a_sign)

        for i in range(len(angle_list)):
            flag1 = 0
            if angle_list[i] >= turn_angle:
                flag1 = 1
            if i > 0 and angle_sign[i] != angle_sign[i - 1]:
                flag1 = 1
            if flag1 == 1:
                seg_index.append(i + 1)
        ##  如果曲线封闭，直接拟合
        if (len(seg_index) - 1) == 0:
            return approxCurve

        for i in range(len(seg_index) - 1):
            seged_edgeset.append(approxCurve[seg_index[i]:seg_index[i + 1] + 1])
        # lineseg1 = np.array(approxCurve[-1]) - np.array(approxCurve[-1])
        # lineseg2 = np.array(approxCurve[0]) - np.array(approxCurve[i])

        # 最后一个点
        lineseg1 = (np.array(approxCurve[-1]) - np.array(approxCurve[-2])).squeeze(0)
        lineseg2 = (np.array(approxCurve[0]) - np.array(approxCurve[-1])).squeeze(0)
        L_1 = np.sqrt(lineseg1.dot(lineseg1))
        L_2 = np.sqrt(lineseg2.dot(lineseg2))

        pi_dist = lineseg1.dot(lineseg2) / (L_1 * L_2)
        if abs(pi_dist) >= 1.:
            # pi_dist = (pi_dist / abs(pi_dist)) * 0.999  # 防止arccos输入超出范围[-1,1]  判断是否超出1
            print('pi_dist value overflow')
        angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi
        a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])  # 不用计算出角度再判断正负，直接用叉乘值判断正负？
        if angle_dist >= turn_angle or a_sign != angle_sign[-1]:
            seged_edgeset.append(approxCurve[seg_index[-1]:])
            seged_edgeset.append(np.concatenate((approxCurve[-1][np.newaxis,], approxCurve[0][np.newaxis,]), 0))
        else:
            seged_edgeset.append(np.concatenate((approxCurve[seg_index[-1]:], approxCurve[0][np.newaxis,]), 0))
        # 判断首末端点连接
        lineseg1 = (np.array(approxCurve[0]) - np.array(approxCurve[-1])).squeeze(0)
        lineseg2 = (np.array(approxCurve[1]) - np.array(approxCurve[0])).squeeze(0)
        L_1 = np.sqrt(lineseg1.dot(lineseg1))
        L_2 = np.sqrt(lineseg2.dot(lineseg2))

        pi_dist = lineseg1.dot(lineseg2) / (L_1 * L_2)
        if abs(pi_dist) >= 1.:
            # pi_dist = (pi_dist / abs(pi_dist)) * 0.999  # 防止arccos输入超出范围[-1,1]  判断是否超出1
            print('pi_dist value overflow')
        angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi
        a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])  # 不用计算出角度再判断正负，直接用叉乘值判断正负？
        if angle_dist <= turn_angle and a_sign == angle_sign[-1]:
            seged_edgeset[0] = np.concatenate((seged_edgeset[-1], seged_edgeset[0]), 0)
            seged_edgeset.pop()
        # if len(seged_edgeset) == 0:
        # 	print('len = 0', len(seged_edgeset))
        seged_edgeset_b = []
        for i in range(len(seged_edgeset)):
            if seged_edgeset[i].shape[0] > 3:
                seged_edgeset_b.append(seged_edgeset[i])
        seged_edgeset = seged_edgeset_b
        # 如果只剩0个曲线 直接返回approcurve前两个值，跳过拟合
        if len(seged_edgeset) == 0:
            print('--------------')
            return approxCurve[0:2]
        # arc 对应contours
        # filted_contour = self.arc2contours(ori_contour,seged_edgeset)
        # print('filted_contour',filted_contour[0].shape)
        # print('filted_contour1',filted_contour[1].shape)
        # print('seged_edgeset', len(seged_edgeset))
        # 如果只剩一个曲线 直接返回arc
        if len(seged_edgeset) == 1:
            return seged_edgeset[0]
        maxarc = 0
        # print('len',len(seged_edgeset))

        #  讲所有弧段旋转顺序调整到同一方向
        for i in range(len(seged_edgeset)):
            if seged_edgeset[i].shape[0] > maxarc:
                max_index = i
                maxarc = seged_edgeset[i].shape[0]
            lineseg1 = (np.array(seged_edgeset[i][2]) - np.array(seged_edgeset[i][1])).squeeze(0)
            lineseg2 = (np.array(seged_edgeset[i][1]) - np.array(seged_edgeset[i][0])).squeeze(0)
            a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])

            if a_sign < 0:
                seged_edgeset[i] = seged_edgeset[i][::-1]

        lineseg1 = (np.array(seged_edgeset[max_index][-1]) - np.array(seged_edgeset[max_index][-2])).squeeze(0)
        lineseg2 = (np.array(seged_edgeset[max_index][1]) - np.array(seged_edgeset[max_index][0])).squeeze(0)
        L_1 = np.sqrt(lineseg1.dot(lineseg1))
        L_2 = np.sqrt(lineseg2.dot(lineseg2))

        pi_dist = lineseg1.dot(lineseg2) / (L_1 * L_2)
        a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])
        if a_sign < 0:
            angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi + 180
        else:
            angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi
        if angle_dist > 270:
            ellip_group.append(seged_edgeset[max_index])
        # ellip_group.append(filted_contour[max_index])
        else:
            ellip_group = self.arc_group(seged_edgeset, max_index)

        for i in range(len(ellip_group)):
            if i == 0:
                arc_Curve = ellip_group[i]
            else:
                arc_Curve = np.concatenate((arc_Curve, ellip_group[i]), 0)
        # ellipse = cv.fitEllipseDirect(arc_Curve)
        return arc_Curve

    def Issearchregion(self, seedarc, subarc):
        # seedarc subarc 形状为三维.l_1等需要转换为一维数据
        l_1 = seedarc[0, 0] - seedarc[1, 0]
        l_2 = seedarc[-1, 0] - seedarc[-2, 0]
        l_m = seedarc[-1, 0] - seedarc[0, 0]
        # sub_midpoint = (subarc[0] + subarc[-1])/2
        sub_midpoint = subarc[subarc.shape[0] // 2 - 1, 0]
        p_t = sub_midpoint - seedarc[0, 0]
        if (l_1[0] * p_t[1] - l_1[1] * p_t[0]) > 0:
            # flat = 0
            return 0
        if (p_t[0] * l_m[1] - p_t[1] * l_m[0]) > 0:
            return 0
        p_t = sub_midpoint - seedarc[-1, 0]
        if (p_t[0] * l_2[1] - p_t[1] * l_2[0]) > 0:
            return 0
        return 1

    def arc_group(self, seged_edgeset, max_index):
        ellip_g = []
        ellip_g.append(seged_edgeset[max_index])
        # ellip_g.append(filted_contour[max_index])
        for i in range(len(seged_edgeset)):
            if i != max_index:
                if self.Issearchregion(seged_edgeset[max_index], seged_edgeset[i]) == 1:
                    if self.Issearchregion(seged_edgeset[i], seged_edgeset[max_index]) == 1:
                        ellip_g.append(seged_edgeset[i])
            # ellip_g.append(filted_contour[i])
        return ellip_g

    def arc2contours(self, contours, arc):
        arc_start = -1
        arc_end = -1
        contour_group = []
        #  先找第一个arc
        for j in range(contours.shape[0]):
            if (contours[j] == arc[0][0]).all():
                arc_start = j
                break
        for j in range(contours.shape[0]):
            if (contours[j] == arc[0][-1]).all():
                arc_end = j
                break
        if arc_start > arc_end:
            contour_one = np.concatenate((contours[arc_start:], contours[0:arc_end + 1]), 0)
            contour_group.append(contour_one)
        else:
            contour_one = contours[arc_start:arc_end]
            contour_group.append(contour_one)
        k = 0
        for i in range(1, len(arc)):
            for j in range(k, contours.shape[0]):
                if (contours[j] == arc[i][0]).all():
                    arc_start = j
                if (contours[j] == arc[i][-1]).all():
                    arc_end = j
                    contour_group.append(contours[arc_start:arc_end])
                    k = j
                    break
        return contour_group

    def cal_draw_ellipse(self, I1, I1_g):
        contours, _ = cv.findContours(cv.medianBlur(I1_g, 15), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        maxindex1 = 0
        maxindex2 = 0
        max1 = 0
        max2 = 0
        # flag1 = 0
        # flag2 = 0
        for j in range(len(contours)):
            if contours[j].shape[0] > max1:
                maxindex1 = j
                max1 = contours[j].shape[0]
            if j == len(contours) - 1:
                approxCurve = cv.approxPolyDP(contours[maxindex1], 2, closed=True)
                if approxCurve.shape[0] > 5:
                    approxCurve = self.edgefliter(approxCurve, turn_angle=90)
                    I1 = cv.drawContours(I1, [contours[maxindex1]], 0, (0, 255, 255), 2)
                    # cv.polylines(I1, [approxCurve], isClosed=False, color=(0, 0, 255), thickness=1, lineType=8, shift=0)
                    # I1 = cv.drawContours(I1, [approxCurve], 0, (0, 0, 255), 1)  # 得到的耻骨联合区域曲线
                    # ellipse = cv.fitEllipse(approxCurve)
                    # cv2.fillPoly(img, contours[1], (255, 0, 0))  # 只染色边界
                    ellipse = cv.fitEllipseDirect(approxCurve)
                    ellipse2 = cv.fitEllipseDirect(contours[maxindex1])
                    cv.ellipse(I1, ellipse, (255, 0, 255), 2)
                    cv.ellipse(I1, ellipse2, (255, 0, 0), 2)
                    cv.polylines(I1, [approxCurve], isClosed=False, color=(0, 0, 255), thickness=3, lineType=8, shift=0)

        return I1

    def video_output(self, video_path, model_name, csv_path):
        unet_path = os.path.join(self.model_path, model_name)

        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

            return

        self.unet.train(False)
        self.unet.eval()
        # writer = SummaryWriter('runs3/aop')

        length = 0
        fps = 20
        size = (512, 384)
        videos_src_path = video_path  # 'D:/py_seg/video7'
        videos = os.listdir(videos_src_path)
        videos = filter(lambda x: x.endswith('mp4'), videos)

        all_video_list = []
        video_name = []
        ####
        for each_video in videos:
            list_aop = []
            print(each_video)
            videowriter = cv.VideoWriter(each_video, cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
            # get the full path of each video, which will open the video tp extract frames
            each_video_full_path = os.path.join(videos_src_path, each_video)

            cap = cv.VideoCapture(each_video_full_path)
            success = True
            frame_num = 0
            while (success):
                success, frame = cap.read()
                frame_num += 1
                if success == False:
                    break
                frame = frame[54:, 528:1823]
                frame[0:434, 1175:] = 0
                frame[959:1003, 199:1048] = 0
                frame[:, 0:72] = 0
                frame = cv.resize(frame, (512, 384))
                # print('Read a new frame: ', success)
                image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 把GRAY图转换为BGR三通道图--BGR转灰度图
                transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(),
                     # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
                     torchvision.transforms.Normalize((0.5,), (0.5,))])

                image = transform(image)
                image = image.unsqueeze(0)
                # image = Norm_(torch.tensor(image)).unsqueeze(0).unsqueeze(0)

                # image = Norm_(image)

                image = image.to(self.device, torch.float)

                # SR, _, _, _, _, _ = self.unet(image)
                SR_lm, SR_seg, _, _, _, SR_cls = self.unet(image)
                SR = torch.softmax(SR_seg, 1)
                SR_cls = torch.softmax(SR_cls, 1)
                SR = SR > 0.5
                SR = SR.mul(255)
                SR = SR.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
                contours, _ = cv.findContours(cv.medianBlur(SR[:, :, 1], 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                contours2, _ = cv.findContours(cv.medianBlur(SR[:, :, 2], 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                img_result = frame
                maxindex1 = 0
                maxindex2 = 0
                max1 = 0
                max2 = 0
                flag1 = 0
                flag2 = 0

                ##########lm
                # lm_h = (SR_lm[:, 0, :, :] + SR_lm[:, 1, :, :]).mul(255)
                # lm_h = lm_h.detach().cpu().numpy().squeeze(0).astype(np.uint8)
                # lm_j = np.zeros((lm_h.shape[0], lm_h.shape[1], 3)).astype(np.uint8)
                # lm_h = self.gray2color(lm_h, jet_map)
                # lm_h = lm_h[:, :, [2, 1, 0]]
                # num = num.cpu().numpy().squeeze(0)
                # image = (image.mul(127)) + 128
                # image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
                # image1 = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                # image_h = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                ########landmark
                SR1 = SR_lm[0, 0, :, :].cpu().detach()
                SR2 = SR_lm[0, 1, :, :].cpu().detach()
                out_cor1 = np.unravel_index(np.argmax(SR1), SR1.shape)
                out_cor2 = np.unravel_index(np.argmax(SR2), SR2.shape)

                for j in range(len(contours)):
                    if contours[j].shape[0] > max1:
                        maxindex1 = j
                        max1 = contours[j].shape[0]
                    if j == len(contours) - 1:
                        approxCurve = cv.approxPolyDP(contours[maxindex1], 2, closed=True)
                        if approxCurve.shape[0] > 5:
                            # img_result = cv.drawContours(img_result, [approxCurve], 0, (0, 0, 255), 1)  # 得到的耻骨联合区域曲线
                            ellipse = cv.fitEllipse(approxCurve)
                            # cv.ellipse(img_result, ellipse, (0, 255, 0), 2)
                            flag1 = 1

                for k in range(len(contours2)):
                    if contours2[k].shape[0] > max2:
                        maxindex2 = k
                        max2 = contours2[k].shape[0]
                    if k == len(contours2) - 1:
                        approxCurve2 = cv.approxPolyDP(contours2[maxindex2], 2, closed=True)
                        if approxCurve2.shape[0] > 5:
                            # img_result = cv.drawContours(img_result, [approxCurve2], 0, (255, 0, 0), 1)
                            ellipse2 = cv.fitEllipse(approxCurve2)
                            cv.ellipse(img_result, ellipse2, (0, 255, 0), 2)
                            flag2 = 1

                if flag1 == 1 and flag2 == 1:
                    # img_result, Aod = drawline_AOD(img_result,ellipse2, ellipse,out_cor2,out_cor1)
                    img_result, Aod = drawline_AOD2(img_result, ellipse2, ellipse)
                    list_aop.append(Aod)

                    cv.putText(img_result, "AOP: " + str(round(Aod, 2)) + ""
                                                                          "", (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 255, 255), 1, cv.LINE_AA)
                    cv.line(img_result, (out_cor1[1] - 4, out_cor1[0]), (out_cor1[1] + 4, out_cor1[0]), (0, 255, 0), 2)
                    cv.line(img_result, (out_cor1[1], out_cor1[0] - 4), (out_cor1[1], out_cor1[0] + 4), (0, 255, 0), 2)

                    cv.line(img_result, (out_cor2[1] - 4, out_cor2[0]), (out_cor2[1] + 4, out_cor2[0]), (0, 255, 0), 2)
                    cv.line(img_result, (out_cor2[1], out_cor2[0] - 4), (out_cor2[1], out_cor2[0] + 4), (0, 255, 0), 2)

                    cv.line(img_result, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)
                    cv.line(img_result, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)

                # writer.add_scalar('AOP/'+each_video,  Aod, frame_num)
                else:
                    # writer.add_scalar('AOP/' + each_video, 0, frame_num)
                    pass
                if SR_cls[0, 1] > 0.5:
                    list_aop.append(Aod)
                else:
                    list_aop.append(0)

                videowriter.write(img_result)

            videowriter.release()
            print('[AOP]  median: %.4f  mean: %.4f  std: %.4f' % (
                np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
            all_video_list.append(list_aop)
            video_name.append(each_video)

        aop_csv = pd.DataFrame(index=video_name, data=all_video_list)
        aop_csv.to_csv(csv_path, encoding='gbk')
        cap.release()

    def test_dc(self):
        # unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
        # unet_path = os.path.join(self.model_path, 'set1-Multi-landmark-MD_UNet-300-0.0001-210-0.5000-512.pkl')
        unet_path =  os.path.join(self.model_path, 'set3_bt2_md-unet-fcn_resnet50-49-0.0001-139-0.3000-512.pkl')
        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')
        # ===================================== test ====================================#
        print('------------------------test------------------------------')
        self.unet.train(False)
        self.unet.eval()
        # self.unet.iter_num = 5
        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        DC_1 = 0.
        DC_2 = 0.
        length = 0

        # for i, (images, GT, GT_Lmark) in enumerate(self.valid_loader):
        # image, GT, GT5, GT_Lmark, cor_num, filename
        with torch.no_grad(): # 测试时候不需要梯度回传 节省空间
            for i, (image, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(self.test_loader):
                # images = self.img_catdist_channel(images)
                images = image.to(self.device)
                # image2 = image2.to(self.device)
                # image3 = image3.to(self.device)
                # image4 = image4.to(self.device)
                GT = GT.to(self.device, torch.long)


                GT = GT.squeeze(1)


                SR_seg = self.unet(images)
                # SR ,_,_,_,_,_= self.unet(images)
                # SR,d1,d2,d3 = self.unet(images)
                # ##多分支网络
                # SR_lm = torch.sigmoid(SR_lm)
                # SR_seg = torch.sigmoid(SR_seg)
                SR_seg = F.softmax(SR_seg, dim=1)
                # print(class_out)
                # class_out = torch.softmax(class_out,dim=1)
                # print('after:',class_out)
                # SR = torch.cat((SR1, SR1, SR2), dim=1)
                # ####
                # SR = F.softmax(SR,dim=1)		#用lovaszloss时不用加sigmoid
                # GT = GT.squeeze(1)
                acc += get_accuracy_test(SR_seg, GT)
                SE += get_sensitivity(SR_seg, GT)
                SP += get_specificity(SR_seg, GT)
                PC += get_precision(SR_seg, GT)
                F1 += get_F1(SR_seg, GT)
                JS += get_JS(SR_seg, GT)
                # DC += get_DC(SR_seg,GT)
                dc_ca0, dc_ca1, dc_ca2 = get_DC_test(SR_seg, GT)
                DC += dc_ca0
                # DC += get_DC(SR_seg, GT)
                DC_1 += dc_ca1
                DC_2 += dc_ca2
                # length += images.size(0)
                length += 1

        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        DC_1 = DC_1 / length
        DC_2 = DC_2 / length

        # unet_score = JS +1.5*DC_1+DC_2

        # class_acc_sum = class_acc.float() / length
        # print('class_acc_sum:',class_acc_sum)
        # unet_score = DC
        # print('class_acc_sum: %.4f' % class_acc_sum)
        # print('[valid-dist] Dist1: %.4f, Dist2: %.4f, Angle_div: %.4f' % (dist_1, dist_2, angle_div))
        # writer.add_scalars('set3_bt2_md-unet-landmark/valid_dist',
        #                    {'dist_1': dist_1, 'dist_2': dist_2, 'angle_div': angle_div}, epoch)
        # writer.add_scalars('set3_bt2_md-unet-landmark/valid_DCgroup', {'DC': DC,
        #                                                                'DC_SH': DC_1,
        #                                                                'DC_Head': DC_2}, epoch)
        print(
            '[test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2))
        info = [acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2]
        dfhistory_test = pd.DataFrame(columns=["acc", "SE", "SP", "PC", "F1", "JS", "DC", "DC1", "DC2"])
        dfhistory_test.loc[1]= info
        dfhistory_test.to_csv('./result/csv/{}_test_result.csv'.format(self.model_type), index=False)

        #--------------
        # print(
        #     '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
        #         acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2))
        # print('[Validation] Dist1: %.4f, Dist2: %.4f,Angle_div: %.4f' % (dist_1, dist_2, angle_div_0))
        # print('[Median] left: %.4f, right: %.4f,Angle: %.4f' % (
        #     np.median(list_l), np.median(list_r), np.median(list_angle)))
        # print('[mean] left: %.4f, right: %.4f,Angle: %.4f' % (np.mean(list_l), np.mean(list_r), np.mean(list_angle)))
        # print('[std] left: %.4f, right: %.4f,Angle: %.4f' % (np.std(list_l), np.std(list_r), np.std(list_angle)))
        # print('[point-num] m_2: %d, m_5: %d,m_10: %d' % (m_2, m_5, m_10))
        # print('[ag-num] total image: %d  ag_5: %d, ag_10: %d' % (length, ag_5, ag_10))

    def test(self):
        # unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
        unet_path = os.path.join(self.model_path, 'set1-Multi-landmark-MD_UNet-300-0.0001-210-0.5000-512.pkl')
        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        # writer = SummaryWriter('runsv2/unet')

        self.unet.train(False)
        self.unet.eval()
        # 读取深度文件
        depth_Path = 'D:/py_seg/Landmark-Net/depth.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        pixel_num = self.image_size * 0.715
        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        DC_1 = 0.
        DC_2 = 0.
        dist_1 = 0
        dist_2 = 0
        angle_div_0 = 0
        m_2 = 0
        m_5 = 0
        m_10 = 0
        ag_5 = 0
        ag_10 = 0
        length = 0
        list_l = []
        list_r = []
        list_angle = []
        for i, (images, GT, GT_Lmark, num) in enumerate(self.valid_loader):  # images, GT, GT_Lmark, GT_class,num

            # images = self.img_catdist_channel(images)
            degth_cm = landmark_depth[num - 1]
            pixel_mm = degth_cm * 10 / pixel_num
            images = images.to(self.device)
            # print(images.shape)
            GT = GT.to(self.device)
            GT_Lmark = GT_Lmark.to(self.device)
            GT = GT.squeeze(1)

            # GT[GT == 1] = 3
            # GT[GT == 2] = 1
            # GT[GT == 3] = 2
            # SR = F.sigmoid(self.unet(images))
            # time1 = time.time()
            # SR = self.unet(images)
            SR_lm, SR_seg, _, _, _, _ = self.unet(images)

            SR_lm = torch.sigmoid(SR_lm)
            SR_seg = torch.softmax(SR_seg, 1)

            # SR1 = F.sigmoid(SR1)
            # SR2 = F.sigmoid(SR2)
            # SR = torch.cat((SR1, SR1, SR2), dim=1)
            # time2 = time.time()
            # SR = F.sigmoid(SR)
            acc += get_accuracy(SR_seg, GT)
            SE += get_sensitivity(SR_seg, GT)
            SP += get_specificity(SR_seg, GT)
            PC += get_precision(SR_seg, GT)
            F1 += get_F1(SR_seg, GT)
            JS += get_JS(SR_seg, GT)
            dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg, GT)
            DC += dc_ca0
            DC_1 += dc_ca1
            DC_2 += dc_ca2
            dist1, dist2, angle_div = get_diatance(SR_lm, GT_Lmark)
            dist1 = pixel_mm * dist1
            dist2 = pixel_mm * dist2
            list_l.append(dist1)
            list_r.append(dist2)
            list_angle.append(angle_div)
            dist_1 += dist1
            dist_2 += dist2
            angle_div_0 += angle_div
            # print(time2-time1)
            length += 1
            print('num: %d -- Dist1: %.4f, Dist2: %.4f,Angle_div: %.4f' % (num, dist1, dist2, angle_div))
            if dist1 <= 2:
                m_2 += 1
            if dist2 <= 2:
                m_2 += 1
            if dist1 <= 5:
                m_5 += 1
            if dist2 <= 5:
                m_5 += 1
            if dist1 <= 10:
                m_10 += 1
            if dist2 <= 10:
                m_10 += 1
            if angle_div <= 5:
                ag_5 += 1
            if angle_div <= 10:
                ag_10 += 1
        # cv.imwrite(r'D:\py_seg\U-Net\U-Net_vari\result\pic_output',SR.numpy())
        # 绘制模型
        # with SummaryWriter(comment='Deform_U_Net') as w:				#其中使用了python的上下文管理，with 语句，可以避免因w.close未写造成的问题
        # 	w.add_graph(self.unet,(images,))
        # 显示原图和特征图可视化
        # img_grid = torchvision.utils.make_grid(images, normalize=True, scale_each=True, nrow=2)
        # # # 绘制原始图像
        # writer.add_image('raw img', img_grid, global_step=666)  # j 表示feature map数
        # #
        # for name, layer in self.unet._modules.items():
        #
        # 	images = layer(images)
        # 	images2 = torch.sum(images, dim=1, keepdim=True)
        # 	print(f'{name}')
        # 	# 第一个卷积没有进行relu，要记得加上
        # 	# x = F.relu(x) if 'Conv' in name else x
        # 	if 'Conv' in name or 'Up' in name:
        # 		# x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
        # 		img_grid = torchvision.utils.make_grid(images2.transpose(0,1), normalize=True, scale_each=True, nrow=8)  # normalize进行归一化处理
        # 		writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)
        # #
        # writer.close()
        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        DC_1 = DC_1 / length
        DC_2 = DC_2 / length
        dist_1 = dist_1 / length
        dist_2 = dist_2 / length
        angle_div_0 = angle_div_0 / length
        print(
            '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2))
        print('[Validation] Dist1: %.4f, Dist2: %.4f,Angle_div: %.4f' % (dist_1, dist_2, angle_div_0))
        print('[Median] left: %.4f, right: %.4f,Angle: %.4f' % (
            np.median(list_l), np.median(list_r), np.median(list_angle)))
        print('[mean] left: %.4f, right: %.4f,Angle: %.4f' % (np.mean(list_l), np.mean(list_r), np.mean(list_angle)))
        print('[std] left: %.4f, right: %.4f,Angle: %.4f' % (np.std(list_l), np.std(list_r), np.std(list_angle)))
        print('[point-num] m_2: %d, m_5: %d,m_10: %d' % (m_2, m_5, m_10))
        print('[ag-num] total image: %d  ag_5: %d, ag_10: %d' % (length, ag_5, ag_10))


    def test_output(self):
        unet_path = os.path.join(self.model_path, 'multitask_use2vgg1d_cls1use.pkl')

        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        self.unet.train(False)
        self.unet.eval()
        # writer = SummaryWriter('runs3/aop')

        length = 0
        fps = 20
        size = (512, 384)
        videos_src_path = 'D:/py_seg/video7'
        videos = os.listdir(videos_src_path)
        videos = filter(lambda x: x.endswith('mp4'), videos)  # 用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象

        all_video_list = []
        video_name = []
        ####
        for each_video in videos:
            list_aop = []
            print(each_video)
            # videowriter = cv.VideoWriter(each_video, cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)  # 视频的写操作
            videowriter = cv.VideoWriter(each_video, cv.VideoWriter_fourcc(*'mp4v'), fps, size)  # 视频的写操作
            # cv.VideoWriter_fourcc 指定写入视频帧编码格式，fps:帧速率
            # get the full path of each video, which will open the video tp extract frames
            each_video_full_path = os.path.join(videos_src_path, each_video)

            cap = cv.VideoCapture(each_video_full_path)  # 视频的读操作
            success = True
            frame_num = 0
            while (success):
                success, frame = cap.read()  # 第二个参数frame表示截取到一帧的图片
                frame_num += 1
                if success == False:
                    break
                frame = frame[54:, 528:1823]
                frame[0:434, 1175:] = 0
                frame[959:1003, 199:1048] = 0
                frame[:, 0:72] = 0
                frame = cv.resize(frame, (512, 384))
                # print('Read a new frame: ', success)
                image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 把GRAY图转换为BGR三通道图--BGR转灰度图
                transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(),
                     # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
                     torchvision.transforms.Normalize((0.5,), (0.5,))])

                image = transform(image)
                image = image.unsqueeze(0)
                # image = Norm_(torch.tensor(image)).unsqueeze(0).unsqueeze(0)

                # image = Norm_(image)
                # print(image.dtype)
                # print(type(image))
                # print(image.shape)
                image = image.to(self.device, torch.float)

                # SR, _, _, _, _, _ = self.unet(image)
                SR_lm, SR_seg, _, _, _, SR_cls = self.unet(image)
                SR = torch.softmax(SR_seg, 1)
                SR_cls = torch.softmax(SR_cls, 1)
                SR = SR > 0.5
                SR = SR.mul(255)
                SR = SR.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
                contours, _ = cv.findContours(cv.medianBlur(SR[:, :, 1], 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                contours2, _ = cv.findContours(cv.medianBlur(SR[:, :, 2], 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                img_result = frame
                maxindex1 = 0
                maxindex2 = 0
                max1 = 0
                max2 = 0
                flag1 = 0
                flag2 = 0

                ##########lm
                # lm_h = (SR_lm[:, 0, :, :] + SR_lm[:, 1, :, :]).mul(255)
                # lm_h = lm_h.detach().cpu().numpy().squeeze(0).astype(np.uint8)
                # lm_j = np.zeros((lm_h.shape[0], lm_h.shape[1], 3)).astype(np.uint8)
                # lm_h = self.gray2color(lm_h, jet_map)
                # lm_h = lm_h[:, :, [2, 1, 0]]
                # num = num.cpu().numpy().squeeze(0)
                # image = (image.mul(127)) + 128
                # image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
                # image1 = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                # image_h = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                ########landmark
                SR1 = SR_lm[0, 0, :, :].cpu().detach()
                SR2 = SR_lm[0, 1, :, :].cpu().detach()
                out_cor1 = np.unravel_index(np.argmax(SR1), SR1.shape)
                out_cor2 = np.unravel_index(np.argmax(SR2), SR2.shape)

                for j in range(len(contours)):
                    if contours[j].shape[0] > max1:
                        maxindex1 = j
                        max1 = contours[j].shape[0]
                    if j == len(contours) - 1:
                        approxCurve = cv.approxPolyDP(contours[maxindex1], 2, closed=True)
                        if approxCurve.shape[0] > 5:
                            # img_result = cv.drawContours(img_result, [approxCurve], 0, (0, 0, 255), 1)  # 得到的耻骨联合区域曲线
                            ellipse = cv.fitEllipse(approxCurve)
                            # cv.ellipse(img_result, ellipse, (0, 255, 0), 2)
                            flag1 = 1

                for k in range(len(contours2)):
                    if contours2[k].shape[0] > max2:
                        maxindex2 = k
                        max2 = contours2[k].shape[0]
                    if k == len(contours2) - 1:
                        approxCurve2 = cv.approxPolyDP(contours2[maxindex2], 2, closed=True)
                        if approxCurve2.shape[0] > 5:
                            # img_result = cv.drawContours(img_result, [approxCurve2], 0, (255, 0, 0), 1)
                            ellipse2 = cv.fitEllipse(approxCurve2)
                            cv.ellipse(img_result, ellipse2, (0, 255, 0), 2)
                            flag2 = 1

                if flag1 == 1 and flag2 == 1:
                    # img_result, Aod = drawline_AOD(img_result,ellipse2, ellipse,out_cor2,out_cor1)
                    img_result, Aod = drawline_AOD2(img_result, ellipse2, ellipse)
                    list_aop.append(Aod)

                    cv.putText(img_result, "AOP: " + str(round(Aod, 2)) + ""
                                                                          "", (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 255, 255), 1, cv.LINE_AA)  # 在图像上绘制文字
                    cv.line(img_result, (out_cor1[1] - 4, out_cor1[0]), (out_cor1[1] + 4, out_cor1[0]), (0, 255, 0), 2)
                    cv.line(img_result, (out_cor1[1], out_cor1[0] - 4), (out_cor1[1], out_cor1[0] + 4), (0, 255, 0), 2)

                    cv.line(img_result, (out_cor2[1] - 4, out_cor2[0]), (out_cor2[1] + 4, out_cor2[0]), (0, 255, 0), 2)
                    cv.line(img_result, (out_cor2[1], out_cor2[0] - 4), (out_cor2[1], out_cor2[0] + 4), (0, 255, 0), 2)

                    cv.line(img_result, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)
                    cv.line(img_result, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)

                # writer.add_scalar('AOP/'+each_video,  Aod, frame_num)
                else:
                    # writer.add_scalar('AOP/' + each_video, 0, frame_num)
                    pass
                if SR_cls[0, 1] > 0.5:
                    list_aop.append(Aod)
                else:
                    list_aop.append(0)

                videowriter.write(img_result)

            videowriter.release()
            print('[AOP]  median: %.4f  mean: %.4f  std: %.4f' % (
                np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
            all_video_list.append(list_aop)
            video_name.append(each_video)

        # aop_csv = pd.DataFrame(index=video_name, data=all_video_list)
        # aop_csv.to_csv(r'D:\py_seg\video7/aop_video_nocls.csv', encoding='gbk')
        cap.release()


def onehot_to_mulchannel(GT):
    print(GT.max())
    for i in range(GT.max() + 1):
        if i == 0:
            GT_sg = GT == i
            print(GT_sg)
            print(GT_sg.shape)
        else:
            GT_sg = torch.cat([GT_sg, GT == i], 1)
            print(GT_sg)
            print(GT_sg.shape)

    return GT_sg.float()

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def get_max_contour(contours):
        max_contour = []
        for contour_item in contours:
            # print(contour_item.shape)
            if len(contour_item) > len(max_contour):
                max_contour = contour_item
        max_contour = [max_contour]  # 填充形状时必须用列表，画轮廓可用可不用
        return max_contour

def test_aop_speed(model_type, test_loader):

    temp_path = os.path.join('./aop_results/{}/unet_image/'.format(model_type)) #为每个模型进行测试并保存
    aop_result_path = temp_path.split('unet_image')[0]
    if  os.path.exists(temp_path) is False:
        os.makedirs(temp_path)
        print('succeed to mkdirs: {}'.format(temp_path))
    time_all = 0
    for i, (images, GT, GT5, GT_Lmark, cor_num, filename) in enumerate(test_loader):

        # time_1 = time.clock()
        # time_1 = time.time()
        time_1 = time.perf_counter()
        # time_1 = time.process_time()
        for index in range(0, len(filename)):
            left_point = right_point = tangent_point = []


            temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], 0))
            # temp_path_item = os.path.join(temp_path, '623-{}-GT-{}.png'.format(filename[0], 0))
            img_background = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)
            img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGR)



            # 获得预测图片的长轴端点和切点
            flag = 0 #图片出错 如全黑 就跳过这个预测
            for i in range(1, 3):
                # temp_path_item = os.path.join(temp_path, 'ATD-{}-real_result-{}.png'.format(cor_num[index], i))
                temp_path_item = os.path.join(temp_path, 'ATD-{}-predict-{}.png'.format(filename[index], i))
                # print(temp_path_item)
                # if os.path.isfile(temp_path_item):
                #     print('yes')
                # else:
                #     print('not')
                img_item = cv2.imread(temp_path_item, cv2.IMREAD_GRAYSCALE)


                _, contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                #contours, hierarchy = cv2.findContours(img_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                max_contour = get_max_contour(contours)
                # print(max_contour)
                if len(max_contour) == 0 or len(max_contour[0]) == 0: # 空白图打印此图片名字，然后跳过这个样例
                    print('picture:{} error, max_contour is {}'.format(temp_path_item, max_contour))
                    flag = 1
                    break
                ellipse_instance = cv2.fitEllipse(max_contour[0])
                cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 1)
                if i == 1:
                    left_point, right_point = get_endpoint(ellipse_instance)
                else:
                    tangent_point = get_tangent_point(right_point, ellipse_instance)

            if flag == 1:
                flag == 0
                continue

            # 计算距离并保存
            # distance = pixel_mm * math.sqrt(math.pow(right_point[0] - left_point[0], 2) + math.pow(right_point[1] - left_point[1], 2))
            cv2.line(img_background, (round(left_point[0]), round(left_point[1])),
                     (round(right_point[0]), round(right_point[1])), (255, 255, 0), 1)
            # cv2.ellipse(img_background, ellipse_instance, (255, 0, 0), 2)
            cv2.line(img_background, (round(right_point[0]), round(right_point[1])),
                     (round(tangent_point[0]), round(tangent_point[1])), (0, 0, 255), 1)
            temp_path_item = os.path.join(temp_path, 'ATD-{}-real_marked_result-{}.png'.format(filename[index], 0))
            cv2.imwrite(temp_path_item, img_background)

            # pic_no = 'ATD-marked_result-{}'.format(filename[index])



            # 计算预测的aop角度 角度计算 a · b = |a| * |b| * cos(o)

            vector_output_axis = (left_point[0] - right_point[0], left_point[1] - right_point[1])

            vector_output_tangent = (tangent_point[0] - right_point[0], tangent_point[1] - right_point[1])

            top = vector_output_axis[0] * vector_output_tangent[0] + vector_output_axis[1] * vector_output_tangent[
                1]
            bottom1 = math.sqrt(math.pow(vector_output_axis[0], 2) + math.pow(vector_output_axis[1], 2))
            bottom2 = math.sqrt(math.pow(vector_output_tangent[0], 2) + math.pow(vector_output_tangent[1], 2))
            output_aop = math.acos(top / (bottom1 * bottom2)) * (180 / math.pi)
        # time_2 = time.clock()
        # time_2 = time.time()
        time_2 = time.perf_counter()
        # time_2 = time.process_time()
        time_all = time_all + time_2 - time_1

    print('{} : aop time: {}s'.format(model_type, time_all))
    print("generate end ......")


if __name__ == '__main__':
    print('1')
    # dfhistory_train = pd.DataFrame(
    #     columns=["epoch", "cost_time", "loss", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "DC1", "DC2"])
    # print(len(dfhistory_train))
    # item1 = (1, '2:3:4', 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    # dfhistory_train.loc[len(dfhistory_train)] = item1
    # print(dfhistory_train)
    # dfhistory_train.to_csv('./test/test.csv', index = False)

    # df.to_csv('./test/test2.csv', index=True)

    # GT = onehot_to_mulchannel(torch.tensor(np.array([[[[0, 0, 1], [1, 2, 2]]], [[[0, 1, 2], [0, 1, 2]]]])))
    # print(GT)
    # GT = GT.squeeze(1)
    # print(GT)
    # print(GT.shape)
    # print(GT[:, 1])
    # print(GT[:, 1].shape)
    # GT = GT[:, 1]
    # print(GT.contiguous().view(GT.shape[0], -1))
    # print(GT.contiguous().view(GT.shape[0], -1).shape)
    ########
    # GT1 = torch.tensor(np.array([[[[0, 1, 0], [1, 0, 0]],
    #                               [[1, 0, 1], [0, 1, 0]],
    #                               [[0, 0, 0], [0, 0, 1]]]]))
    #
    # GT2 = torch.tensor(np.array([[[[2, 0, 1], [1, 2, 2]]]]))
    # GT2 = torch.tensor(np.array([[[[1, 0, 1], [0, 1, 2]]]]))
    # GT2 = GT2.squeeze(1)
    # print(get_sensitivity(GT1, GT2))
