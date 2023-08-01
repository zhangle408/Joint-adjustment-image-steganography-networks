# encoding: utf-8

import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import Softmax

# import utils.transformed as transforms
#from torchvision import transforms
# from data.ImageFolderDataset import MyImageFolder
#from models.HidingUNet_C1_1 import UnetGenerator_C
#from models.HidingUNet_S1_1 import UnetGenerator_S
from models.HidingUNet import UnetGenerator
from models.RevealNet_S import RevealNet_S
from models.RevealNet_C import RevealNet_C
from torchvision.datasets import ImageFolder
import pdb
import math
import random
import numpy as np
import cv2
from skimage.measure import compare_ssim as SSIM, compare_psnr as PSNR
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.identity import Identity

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
# parser.add_argument('--batchSize', type=int, default=48,
#                     help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--epochs', type=int, default=65,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2,
                    help='number of GPUs to use')
parser.add_argument('--Rnet_C', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet_S', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Hnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--alpha', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--gama', type=float, default=0.75,
                    help='hyper parameter of gama')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='checkpoint folder')
parser.add_argument('--test_diff', default='', help='another checkpoint folder')
parser.add_argument('--checkpoint', default='', help='checkpoint address')
parser.add_argument('--checkpoint_diff', default='', help='another checkpoint address')

parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=1000, help='the frequency of save the resultPic')
parser.add_argument('--norm', default='instance', help='batch or instance')
parser.add_argument('--loss', default='l2', help='l1 or l2')
parser.add_argument('--num_secret', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--num_cover', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--bs_secret', type=int, default=32, help='batch size for ')
parser.add_argument('--num_training', type=int, default=1,
                    help='During training, how many cover images are used for one secret image')
parser.add_argument('--channel_cover', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--channel_secret', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--iters_per_epoch', type=int, default=2000, help='1: gray; 3: color')
parser.add_argument('--no_cover', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--plain_cover', type=bool, default=False, help='use plain cover')
parser.add_argument('--noise_cover', type=bool, default=False, help='use noise cover')
parser.add_argument('--cover_dependent', type=bool, default=False,
                    help='Whether the secret image is dependent on the cover image')
parser.add_argument('--n', type=int, default=2,
                    help='number of the IRU')


# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


# Print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log(str(net),'Total number of parameters: %d' % num_params, logPath)


# Code saving
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)
    cur_work_dir, mainfile = os.path.split(main_file_path)

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


def main():
    ############### Define global parameters ###############
    global opt, optimizer, optimizerR, writer, logPath, scheduler, schedulerR, val_loader, smallestLoss, DATA_DIR,noiser_dropout, noiser_gaussian, noiser_identity

    opt = parser.parse_args()
    #opt.ngpu = torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    '''if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")'''

    cudnn.benchmark = True

    '''if opt.hostname == 'DL178':
        DATA_DIR = '/media/user/SSD1TB-2/ImageNet' 
    assert DATA_DIR'''

    ############  Create the dirs to save the result ############
    if not opt.debug:
        try:
            cur_time = time.strftime('%Y-%m-%d_H%H-%M-%S', time.localtime())
            if opt.test == '':
                secret_comment = 'color' if opt.channel_secret == 3 else 'gray'
                cover_comment = 'color' if opt.channel_cover == 3 else 'gray'
                comment = str(opt.num_secret) + secret_comment + 'In' + str(opt.num_cover) + cover_comment
                '''experiment_dir = opt.hostname + "_" + cur_time + "_" + str(opt.imageSize) + "_" + str(
                    opt.num_secret) + "_" + str(opt.num_training) + "_" + \
                                 str(opt.bs_secret) + "_" + "_" + opt.norm + "_" + opt.loss + "_" + str(
                    opt.beta) + "_" + comment + "_" + opt.remark'''
                experiment_dir = opt.remark + "_" + cur_time
                opt.outckpts += experiment_dir + "/checkPoints"
                opt.trainpics += experiment_dir + "/trainPics"
                opt.validationpics += experiment_dir + "/validationPics"
                opt.outlogs += experiment_dir + "/trainingLogs"
                opt.outcodes += experiment_dir + "/codes"
                if not os.path.exists(opt.outckpts):
                    os.makedirs(opt.outckpts)
                if not os.path.exists(opt.trainpics):
                    os.makedirs(opt.trainpics)
                if not os.path.exists(opt.validationpics):
                    os.makedirs(opt.validationpics)
                if not os.path.exists(opt.outlogs):
                    os.makedirs(opt.outlogs)
                if not os.path.exists(opt.outcodes):
                    os.makedirs(opt.outcodes)
                save_current_codes(opt.outcodes)
            else:
                experiment_dir = opt.test
                opt.testPics += experiment_dir + "/testPics"
                opt.validationpics = opt.testPics
                opt.outlogs += experiment_dir + "/testLogs"
                if (not os.path.exists(opt.testPics)) and opt.test != '':
                    os.makedirs(opt.testPics)
                if not os.path.exists(opt.outlogs):
                    os.makedirs(opt.outlogs)
        except OSError:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")  # ignore

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.bs_secret)
    if opt.debug:
        logPath = './debug/debug_logs/debug.txt'
    print_log(str(opt), logPath)

    ##################  Datasets  ##################
    # traindir = os.path.join(DATA_DIR, 'train')
    # valdir = os.path.join(DATA_DIR, 'val')
    traindir = os.path.join('/data/data/Imagenet2012/imagenet2012_train')
    valdir = os.path.join('/data/data/Imagenet2012/imagenet2012_val')

    transforms_color = transforms.Compose([
        transforms.Resize([opt.imageSize, opt.imageSize]),
        transforms.ToTensor(),
    ])

    transforms_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([opt.imageSize, opt.imageSize]),
        transforms.ToTensor(),
    ])
    if opt.channel_cover == 1:
        transforms_cover = transforms_gray
    else:
        transforms_cover = transforms_color

    if opt.channel_secret == 1:
        transforms_secret = transforms_gray
    else:
        transforms_secret = transforms_color

    if opt.test == '':
        train_dataset_cover = ImageFolder(
            traindir,
            transforms_cover)

        train_dataset_secret = ImageFolder(
            traindir,
            transforms_secret)

        val_dataset_cover = ImageFolder(
            valdir,
            transforms_cover)
        val_dataset_secret = ImageFolder(
            valdir,
            transforms_secret)

        assert train_dataset_cover;
        assert train_dataset_secret
        assert val_dataset_cover;
        assert val_dataset_secret
    else:
        opt.checkpoint = "./training/" + opt.test + "/checkPoints/" + "checkpoint.pth.tar"
        if opt.test_diff != '':
            opt.checkpoint_diff = "./training/" + opt.test_diff + "/checkPoints/" + "checkpoint.pth.tar"
        testdir = valdir
        test_dataset_cover = ImageFolder(
            testdir,
            transforms_cover)
        test_dataset_secret = ImageFolder(
            testdir,
            transforms_secret)
        assert test_dataset_cover;
        assert test_dataset_secret

    ##################  Hiding and Reveal  ##################
    assert opt.imageSize % 32 == 0
    num_downs = 5
    if opt.norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    if opt.norm == 'batch':
        norm_layer = nn.BatchNorm2d
    if opt.norm == 'none':
        norm_layer = None
    '''norm_layer_H = nn.InstanceNorm2d
    norm_layer_R = nn.BatchNorm2d'''
    if opt.cover_dependent:
        Hnet_1 = UnetGenerator(input_nc=opt.channel_secret * opt.num_secret + opt.channel_cover * opt.num_cover,
                             output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs, norm_layer=norm_layer,
                             use_tanh=0)
        Hnet_2 = UnetGenerator(input_nc=opt.channel_secret * opt.num_secret + opt.channel_cover * opt.num_cover,
                               output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs, norm_layer=norm_layer,
                               use_tanh=0)
    else:
        #print('opt.beta',opt.beta)
        '''Hnet_C = UnetGenerator_C(input_nc=opt.channel_secret * opt.num_cover, output_nc=opt.channel_cover * opt.num_cover,
                             num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Tanh)
        Hnet_S = UnetGenerator_S(input_nc=opt.channel_secret * opt.num_secret,
                                 output_nc=opt.channel_cover * opt.num_cover,
                                 num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Tanh)'''
        Hnet_1 = UnetGenerator(input_nc=opt.channel_secret * opt.num_secret,
                             output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs, norm_layer=norm_layer,
                             use_tanh=1)
        Hnet_2 = UnetGenerator(input_nc=opt.channel_secret * opt.num_secret,
                               output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs, norm_layer=norm_layer,
                               use_tanh=1)
    Rnet_S_1 = RevealNet_S(input_nc=opt.channel_cover * opt.num_cover, output_nc=opt.channel_secret * opt.num_secret, nhf=64,
                     norm_layer=norm_layer, output_function=nn.Sigmoid)
    Rnet_C_1 = RevealNet_C(input_nc=opt.channel_cover * opt.num_cover, output_nc=opt.channel_cover * opt.num_cover,
                         nhf=64,
                         norm_layer=norm_layer, output_function=nn.Sigmoid)
    Rnet_S_2 = RevealNet_S(input_nc=opt.channel_cover * opt.num_cover, output_nc=opt.channel_secret * opt.num_secret,
                           nhf=64,
                           norm_layer=norm_layer, output_function=nn.Sigmoid)
    Rnet_C_2 = RevealNet_C(input_nc=opt.channel_cover * opt.num_cover, output_nc=opt.channel_cover * opt.num_cover,
                           nhf=64,
                           norm_layer=norm_layer, output_function=nn.Sigmoid)
    print('opt.cover_depent',opt.cover_dependent)
    print('opt.n', opt.n)
    print('opt.alpha', opt.alpha)
    print('opt.beta', opt.beta)
    print('opt.gama', opt.gama)
    p = 0.3
    noiser_dropout = Dropout([p, p])
    noiser_gaussian = gaussian_kernel()
    noiser_identity = Identity()

    if opt.cover_dependent:
        assert opt.num_training == 1
        assert opt.no_cover == False

    ##### Always set to multiple GPU mode  #####
    Rnet_C_1 = torch.nn.DataParallel(Rnet_C_1).cuda()
    Rnet_S_1 = torch.nn.DataParallel(Rnet_S_1).cuda()
    Hnet_1 = torch.nn.DataParallel(Hnet_1).cuda()
    Rnet_C_2 = torch.nn.DataParallel(Rnet_C_2).cuda()
    Rnet_S_2 = torch.nn.DataParallel(Rnet_S_2).cuda()
    Hnet_2 = torch.nn.DataParallel(Hnet_2).cuda()
    noiser_dropout = torch.nn.DataParallel(noiser_dropout).cuda()
    noiser_gaussian = torch.nn.DataParallel(noiser_gaussian).cuda()
    noiser_identity = torch.nn.DataParallel(noiser_identity).cuda()


    if opt.checkpoint != "":
        checkpoint = torch.load(opt.checkpoint)
        Rnet_C_1.load_state_dict(checkpoint['R_C_1_state_dict'])
        Rnet_S_1.load_state_dict(checkpoint['R_S_1_state_dict'])
        Hnet_1.load_state_dict(checkpoint['H_1_state_dict'])
        Rnet_C_2.load_state_dict(checkpoint['R_C_2_state_dict'])
        Rnet_S_2.load_state_dict(checkpoint['R_S_2_state_dict'])
        Hnet_2.load_state_dict(checkpoint['H_2_state_dict'])

    # Loss and Metric
    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    # Train the networks when opt.test is empty
    if opt.test == '':
        if not opt.debug:
            writer = SummaryWriter(log_dir='runs/' + experiment_dir)
        params = list(Rnet_C_1.parameters()) + list(Hnet_1.parameters()) + list(Rnet_S_1.parameters()) + list(Rnet_C_2.parameters()) + list(Hnet_2.parameters()) + list(Rnet_S_2.parameters())
        optimizer = optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)

        train_loader_secret = DataLoader(train_dataset_secret, batch_size=opt.bs_secret * opt.num_secret,
                                         shuffle=True, num_workers=int(opt.workers))
        train_loader_cover = DataLoader(train_dataset_cover,
                                        batch_size=opt.bs_secret * opt.num_cover * opt.num_training,
                                        shuffle=True, num_workers=int(opt.workers))
        val_loader_secret = DataLoader(val_dataset_secret, batch_size=opt.bs_secret * opt.num_secret,
                                       shuffle=False, num_workers=int(opt.workers))
        val_loader_cover = DataLoader(val_dataset_cover, batch_size=opt.bs_secret * opt.num_cover * opt.num_training,
                                      shuffle=True, num_workers=int(opt.workers))

        smallestLoss = 10000
        print_log("training is beginning .......................................................", logPath)
        for epoch in range(opt.epochs):
            adjust_learning_rate(optimizer, epoch)
            train_loader = zip(train_loader_secret, train_loader_cover)
            val_loader = zip(val_loader_secret, val_loader_cover)

            ######################## train ##########################################
            #print('traing************')
            train(train_loader, epoch, Rnet_C_1=Rnet_C_1, Rnet_S_1=Rnet_S_1, Hnet_1=Hnet_1, Rnet_C_2=Rnet_C_2, Rnet_S_2=Rnet_S_2, Hnet_2=Hnet_2, criterion=criterion)

            ####################### validation  #####################################
            val_hloss, val_rloss, val_hdiff, val_rdiff = validation(val_loader, epoch, Rnet_C_1=Rnet_C_1, Rnet_S_1=Rnet_S_1, Hnet_1=Hnet_1, Rnet_C_2=Rnet_C_2, Rnet_S_2=Rnet_S_2, Hnet_2=Hnet_2,
                                                                    criterion=criterion)

            ####################### adjust learning rate ############################
            scheduler.step(val_rloss)

            # Save the best model parameters
            sum_diff = val_hdiff + val_rdiff
            is_best = sum_diff < globals()["smallestLoss"]
            globals()["smallestLoss"] = sum_diff

            save_checkpoint({
                'epoch': epoch + 1,
                'R_C_1_state_dict': Rnet_C_1.state_dict(),
                'R_S_1_state_dict': Rnet_S_1.state_dict(),
                'H_1_state_dict': Hnet_1.state_dict(),
                'R_C_2_state_dict': Rnet_C_2.state_dict(),
                'R_S_2_state_dict': Rnet_S_2.state_dict(),
                'H_2_state_dict': Hnet_2.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch, '%s/epoch_%d_Hloss_%.4f_Rloss=%.4f_Hdiff_Hdiff%.4f_Rdiff%.4f' % (
            opt.outckpts, epoch, val_hloss, val_rloss, val_hdiff, val_rdiff))

        if not opt.debug:
            writer.close()

    # For testing the trained network
    else:
        test_loader_secret = DataLoader(test_dataset_secret, batch_size=opt.bs_secret * opt.num_secret,
                                        shuffle=True, num_workers=int(opt.workers))
        test_loader_cover = DataLoader(test_dataset_cover, batch_size=opt.bs_secret * opt.num_cover * opt.num_training,
                                       shuffle=True, num_workers=int(opt.workers))
        test_loader = zip(test_loader_secret, test_loader_cover)
        # validation(test_loader, 0, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
        #analysis(test_loader, 0, Hnet=Hnet, Rnet=Rnet, HnetD=HnetD, RnetD=RnetD, criterion=criterion)
        analysis(test_loader, 0, Rnet_C_1=Rnet_C_1, Rnet_S_1=Rnet_S_1, Hnet_1=Hnet_1, Rnet_C_2=Rnet_C_2, Rnet_S_2=Rnet_S_2, Hnet_2=Hnet_2,criterion=criterion)
        #analysis_n(test_loader, 0, Rnet_C=Rnet_C, Rnet_S=Rnet_S, Hnet=Hnet, criterion=criterion)
        #analysis_origion(test_loader, 0, Hnet_C=Hnet_C, Hnet_S=Hnet_S, Rnet=Rnet, criterion=criterion)


def save_checkpoint(state, is_best, epoch, prefix):
    filename = '%s/checkpoint.pth.tar' % opt.outckpts

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/best_checkpoint.pth.tar' % opt.outckpts)
    if epoch == opt.epochs - 1:
        with open(opt.outckpts + prefix + '.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            # writer.writerow([epoch, loss, train1, train5, prec1, prec5])


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=3, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x
attention = CrissCrossAttention(in_dim=3)
attention = torch.nn.DataParallel(attention).cuda()
def forward_pass_1(secret_img, secret_target, cover_img, cover_target, Rnet_C_1, Rnet_S_1, Hnet_1, deta_container, val_cover=0, i_c=None,
                 position=None, Se_two=None):
    batch_size_secret, channel_secret, h_secret, w_secret = secret_img.size()
    batch_size_cover, channel_cover, h_cover, w_cover = cover_img.size()

    # Put tensors in GPU
    if opt.cuda:
        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()
        deta_container = deta_container.cuda()
        # concat_img = concat_img.cuda()

    '''secret_imgv = secret_img.view(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret, opt.imageSize,
                                  opt.imageSize)
    secret_imgv_nh = secret_imgv.repeat(opt.num_training, 1, 1, 1)

    cover_img = cover_img.view(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                               opt.imageSize)'''
    secret_imgv = secret_img
    secret_imgv_nh = secret_imgv
    cover_img = cover_img

    if opt.no_cover and (
            val_cover == 0):  # if val_cover = 1, always use cover in val; otherwise, no_cover True >>> not using cover in training
        cover_img.fill_(0.0)
        print('no_cover')
    if (opt.plain_cover or opt.noise_cover) and (val_cover == 0):
        cover_img.fill_(0.0)
        print('plain_cover')
    b, c, w, h = cover_img.size()

    if opt.plain_cover and (val_cover == 0):
        img_w1 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_w2 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_w3 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_w4 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_wh = torch.cat((img_w1, img_w2, img_w3, img_w4), dim=3)
        cover_img = cover_img + img_wh
        print('if opt.plain_cover and (val_cover == 0):')
    if opt.noise_cover and (val_cover == 0):
        cover_img = cover_img + ((torch.rand(b, c, w, h) - 0.5) * 2 * 0 / 255).cuda()
        print('if opt.noise_cover and (val_cover == 0):')
#+++++++++++++++++++++++++++
    cover_imgv = cover_img
    if opt.cover_dependent:
        H_input = torch.cat((cover_imgv, secret_imgv), dim=1)
    else:
        H_input = secret_imgv
    # ********************************************
    itm_secret_img = Hnet_1(H_input)

    #out_dct_1, out_dct_2, out_dct_3, out_dct_4, itm_secret_img = Hnet_S(H_input)
    #**************
    if i_c != None:
        print('if i_c != None')
        if type(i_c) == type(1.0):
            ####### To keep one channel #######
            itm_secret_img_clone = itm_secret_img.clone()
            itm_secret_img.fill_(0)
            itm_secret_img[:, int(i_c):int(i_c) + 1, :, :] = itm_secret_img_clone[:, int(i_c):int(i_c) + 1, :, :]
        if type(i_c) == type(1):
            print('aaaaa', i_c)
            ####### To set one channel to zero #######
            itm_secret_img[:, i_c:i_c + 1, :, :].fill_(0.0)

    if position != None:
        print('if position != None')
        itm_secret_img[:, :, position:position + 1, position:position + 1].fill_(0.0)
    if Se_two == 2:
        print('if Se_two == 2')
        itm_secret_img_half = itm_secret_img[0:batch_size_secret // 2, :, :, :]
        itm_secret_img = itm_secret_img + torch.cat((itm_secret_img_half.clone().fill_(0.0), itm_secret_img_half), 0)
    elif type(Se_two) == type(0.1):
        print('type(Se_two) == type(0.1)')
        itm_secret_img = itm_secret_img + Se_two * torch.rand(itm_secret_img.size()).cuda()
    if opt.cover_dependent:
        container_img = itm_secret_img
    else:
        #print('opt.num_training', opt.num_training)
        itm_secret_img = itm_secret_img.repeat(opt.num_training, 1, 1, 1)
        #print(itm_secret_img.shape)
        #***************
        container_img = itm_secret_img + cover_imgv
        #container_img = Hnet_C(cover_img, out_dct_1, out_dct_2, out_dct_3, out_dct_4, itm_secret_img)
        #**************
    #new_container_img = container_img
    #print('deta_container.shape', deta_container.shape)
    #print('container_img.shape', container_img.shape)
    #new_container_img = attention(deta_container)+attention(container_img)
    #new_container_img = deta_container + container_img
    #errH = criterion(container_img, cover_imgv)  # Hiding net

    #rev_secret_img = Rnet(container_img, S_S)
    #print(new_container_img.shape)
    rev_secret_img = Rnet_S_1(container_img)
    rev_cover_img = Rnet_C_1(container_img)
    #errR = criterion(rev_secret_img, secret_imgv_nh)  # Reveal net
    #errRC = criterion(rev_cover_img, container_img)  # Reveal net

    # L1 metric
    #diffH = (container_img - cover_imgv).abs().mean() * 255
    #diffR = (rev_secret_img - secret_imgv_nh).abs().mean() * 255
    #diffRC = (rev_cover_img - container_img).abs().mean() * 255
    return cover_imgv, container_img, secret_imgv_nh, rev_secret_img, rev_cover_img#, errH, errR, errRC, diffH, diffR, diffRC
def forward_pass_2(secret_img, secret_target, cover_img, cover_target, Rnet_C_2, Rnet_S_2, Hnet_2, deta_container, val_cover=0, i_c=None,
                 position=None, Se_two=None):
    batch_size_secret, channel_secret, h_secret, w_secret = secret_img.size()
    batch_size_cover, channel_cover, h_cover, w_cover = cover_img.size()

    # Put tensors in GPU
    if opt.cuda:
        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()
        deta_container = deta_container.cuda()
        # concat_img = concat_img.cuda()

    '''secret_imgv = secret_img.view(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret, opt.imageSize,
                                  opt.imageSize)
    secret_imgv_nh = secret_imgv.repeat(opt.num_training, 1, 1, 1)

    cover_img = cover_img.view(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                               opt.imageSize)'''
    secret_imgv = secret_img
    secret_imgv_nh = secret_imgv
    cover_img = cover_img

    if opt.no_cover and (
            val_cover == 0):  # if val_cover = 1, always use cover in val; otherwise, no_cover True >>> not using cover in training
        cover_img.fill_(0.0)
        print('no_cover')
    if (opt.plain_cover or opt.noise_cover) and (val_cover == 0):
        cover_img.fill_(0.0)
        print('plain_cover')
    b, c, w, h = cover_img.size()

    if opt.plain_cover and (val_cover == 0):
        img_w1 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_w2 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_w3 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_w4 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_wh = torch.cat((img_w1, img_w2, img_w3, img_w4), dim=3)
        cover_img = cover_img + img_wh
        print('if opt.plain_cover and (val_cover == 0):')
    if opt.noise_cover and (val_cover == 0):
        cover_img = cover_img + ((torch.rand(b, c, w, h) - 0.5) * 2 * 0 / 255).cuda()
        print('if opt.noise_cover and (val_cover == 0):')
#+++++++++++++++++++++++++++
    cover_imgv = cover_img
    if opt.cover_dependent:
        H_input = torch.cat((cover_imgv, secret_imgv), dim=1)
    else:
        H_input = secret_imgv
    # ********************************************
    itm_secret_img = Hnet_2(H_input)

    #out_dct_1, out_dct_2, out_dct_3, out_dct_4, itm_secret_img = Hnet_S(H_input)
    #**************
    if i_c != None:
        print('if i_c != None')
        if type(i_c) == type(1.0):
            ####### To keep one channel #######
            itm_secret_img_clone = itm_secret_img.clone()
            itm_secret_img.fill_(0)
            itm_secret_img[:, int(i_c):int(i_c) + 1, :, :] = itm_secret_img_clone[:, int(i_c):int(i_c) + 1, :, :]
        if type(i_c) == type(1):
            print('aaaaa', i_c)
            ####### To set one channel to zero #######
            itm_secret_img[:, i_c:i_c + 1, :, :].fill_(0.0)

    if position != None:
        print('if position != None')
        itm_secret_img[:, :, position:position + 1, position:position + 1].fill_(0.0)
    if Se_two == 2:
        print('if Se_two == 2')
        itm_secret_img_half = itm_secret_img[0:batch_size_secret // 2, :, :, :]
        itm_secret_img = itm_secret_img + torch.cat((itm_secret_img_half.clone().fill_(0.0), itm_secret_img_half), 0)
    elif type(Se_two) == type(0.1):
        print('type(Se_two) == type(0.1)')
        itm_secret_img = itm_secret_img + Se_two * torch.rand(itm_secret_img.size()).cuda()
    if opt.cover_dependent:
        container_img = itm_secret_img
    else:
        #print('opt.num_training', opt.num_training)
        itm_secret_img = itm_secret_img.repeat(opt.num_training, 1, 1, 1)
        #print(itm_secret_img.shape)
        #***************
        container_img = itm_secret_img + cover_imgv
        #container_img = Hnet_C(cover_img, out_dct_1, out_dct_2, out_dct_3, out_dct_4, itm_secret_img)
        #**************
    #new_container_img = container_img
    #print('deta_container.shape', deta_container.shape)
    #print('container_img.shape', container_img.shape)
    #new_container_img = attention(deta_container)+attention(container_img)
    new_container_img = deta_container + container_img
    #errH = criterion(container_img, cover_imgv)  # Hiding net

    #rev_secret_img = Rnet(container_img, S_S)
    #print(new_container_img.shape)
    rev_secret_img = Rnet_S_2(new_container_img)
    rev_cover_img = Rnet_C_2(new_container_img)
    #errR = criterion(rev_secret_img, secret_imgv_nh)  # Reveal net
    #errRC = criterion(rev_cover_img, container_img)  # Reveal net

    # L1 metric
    #diffH = (container_img - cover_imgv).abs().mean() * 255
    #diffR = (rev_secret_img - secret_imgv_nh).abs().mean() * 255
    #diffRC = (rev_cover_img - container_img).abs().mean() * 255
    return cover_imgv, container_img, new_container_img, secret_imgv_nh, rev_secret_img, rev_cover_img#, errH, errR, errRC, diffH, diffR, diffRC


class deta_conv(nn.Module):
    def __init__(self,input_c, output_c):
        super(deta_conv, self).__init__()
        self.conv = nn.Conv2d(input_c, output_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.conv(x))

        #return x * y.expand_as(x)
        return out
deta_conv_1 = deta_conv(input_c=3, output_c=3)
deta_conv_1 = torch.nn.DataParallel(deta_conv_1).cuda()
class container_conv(nn.Module):
    def __init__(self, input_c, output_c):
        super(container_conv, self).__init__()
        self.conv = nn.Conv2d(input_c, output_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(output_c)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = (self.sigmoid(self.conv(x)))

        # return x * y.expand_as(x)
        return out
container_conv_1 = container_conv(input_c=6, output_c=3)
container_conv_1 = torch.nn.DataParallel(container_conv_1).cuda()



def train(train_loader, epoch, Rnet_C_1, Rnet_S_1, Hnet_1, Rnet_C_2, Rnet_S_2, Hnet_2,  criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    RClosses = AverageMeter()
    SumLosses = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    RCdiff = AverageMeter()

    # Switch to train mode
    Rnet_C_1.train()
    Rnet_S_1.train()
    Hnet_1.train()
    Rnet_C_2.train()
    Rnet_S_2.train()
    Hnet_2.train()

    start_time = time.time()

    for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(train_loader, 0):

        data_time.update(time.time() - start_time)

        '''cover_imgv, container_img, secret_imgv_nh, rev_secret_img, rev_cover_img, errH, errR, errRC, diffH, diffR, diffRC \
            = forward_pass(secret_img, secret_target, cover_img, cover_target, Rnet_C, Rnet_S, Hnet, criterion)'''
        '''secret_imgv = secret_img.view(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret, opt.imageSize,
                                  opt.imageSize)
    secret_imgv_nh = secret_imgv.repeat(opt.num_training, 1, 1, 1)

    cover_img = cover_img.view(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                               opt.imageSize)'''
        #deta_secret = secret_img.cuda()
        batch_size_secret, channel_secret, h, w = secret_img.size()
        batch_size_cover, channel_cover, h_s, w_s = cover_img.size()

        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()
        #print('cover_img', cover_img)
        #print('secret_img', secret_img)

        deta_secret = secret_img.view(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret, opt.imageSize,
                                  opt.imageSize).repeat(opt.num_training, 1, 1, 1)
        deta_cover = cover_img.view(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                               opt.imageSize)
        deta_container = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                               opt.imageSize).to(cover_img.device)

        sum_container = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                               opt.imageSize).to(cover_img.device)
        last_container_img= torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                               opt.imageSize).to(cover_img.device)
        sum_cover = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                               opt.imageSize).to(cover_img.device)
        sum_secret = torch.zeros(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret, opt.imageSize,
                                  opt.imageSize).repeat(opt.num_training, 1, 1, 1).to(cover_img.device)
        cover_imgv_1, container_img_1, secret_imgv_nh_1, rev_secret_img_1, rev_cover_img_1 \
            = forward_pass_1(deta_secret, secret_target, deta_cover, cover_target, Rnet_C_1, Rnet_S_1, Hnet_1, deta_container)
        res_cover = (cover_img - rev_cover_img_1)
        res_secret = (secret_img - rev_secret_img_1)
        deta_cover_1 = res_cover.abs()
        deta_secret_1 = res_secret.abs()
        #deta_cover_1 = rev_cover_img_1
        #print('deta_secret.shape', deta_secret.shape)
        #print('rev_secret_img_1.shape', rev_secret_img_1.shape)
        #deta_secret_1 = torch.cat([deta_secret, rev_secret_img_1], 1)
        deta_container_1 = container_img_1
        cover_imgv_2, container_img_2, new_container_img_2, secret_imgv_nh_2, rev_secret_img_2, rev_cover_img_2 \
            = forward_pass_2(deta_secret_1, secret_target, deta_cover_1, cover_target, Rnet_C_2, Rnet_S_2, Hnet_2,
                           deta_container_1)

        errH = criterion(new_container_img_2, cover_img)  # Hiding net container
        errR = criterion(rev_secret_img_2, secret_img)  # Reveal net secret
        errRC = criterion(rev_cover_img_2, cover_img)  # Reveal net cover

        # L1 metric
        diffH = (new_container_img_2 - cover_img).abs().mean() * 255  ## Hiding net container
        diffR = (rev_secret_img_2 - secret_img).abs().mean() * 255  # Reveal net secret
        diffRC = (rev_cover_img_2 - cover_img).abs().mean() * 255  # Reveal net cover


        Hlosses.update(errH.item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        Rlosses.update(errR.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        RClosses.update(errRC.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        Hdiff.update(diffH.item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Rdiff.update(diffR.item(), opt.bs_secret * opt.num_secret * opt.num_training)
        RCdiff.update(diffRC.item(), opt.bs_secret * opt.num_secret * opt.num_training)

        # Loss, backprop, and optimization step
        gamaeerR_container = opt.gama * errRC
        betaerrR_secret = opt.beta * errR
        err_sum = opt.alpha*errH + betaerrR_secret + gamaeerR_container
        optimizer.zero_grad()
        err_sum.backward()
        optimizer.step()

        # Time spent on one batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\tLoss_H: %.6f Loss_R: %.6f Loss_RC: %.6f L1_H: %.4f L1_R: %.4f L1_RC: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.epochs, i, opt.iters_per_epoch,
            Hlosses.val, Rlosses.val, RClosses.val, Hdiff.val, Rdiff.val, RCdiff.val, data_time.val, batch_time.val)

        if i % opt.logFrequency == 0:
            print(log)

        '''if epoch == opt.epochs and i % opt.resultPicFrequency == 0:
            save_result_pic(opt.bs_secret * opt.num_training, cover_imgv, container_img.data, secret_imgv_nh,
                            rev_secret_img.data, epoch, i, opt.trainpics)'''

        if i == opt.iters_per_epoch - 1:
            break

    # To save the last batch only
    '''save_result_pic(opt.bs_secret * opt.num_training, cover_imgv, container_img.data, secret_imgv_nh,
                    rev_secret_img.data, epoch, i, opt.trainpics)'''

    epoch_log = "Training[%d] Hloss=%.6f\tRloss=%.6f\tRCloss=%.6f\tHdiff=%.4f\tRdiff=%.4f\tRCdiff=%.4f\tlr= %.6f\t Epoch time= %.4f" % (
    epoch, Hlosses.avg, Rlosses.avg, RClosses.avg, Hdiff.avg, Rdiff.avg, RCdiff.avg, optimizer.param_groups[0]['lr'], batch_time.sum)
    print_log(epoch_log, logPath)

    if not opt.debug:
        writer.add_scalar("lr/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
        writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
        writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
        writer.add_scalar('train/RC_loss', RClosses.avg, epoch)
        writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)
        writer.add_scalar('train/H_diff', Hdiff.avg, epoch)
        writer.add_scalar('train/R_diff', Rdiff.avg, epoch)
        writer.add_scalar('train/RC_diff', RCdiff.avg, epoch)


def validation(val_loader, epoch, Rnet_C_1, Rnet_S_1,Hnet_1, Rnet_C_2, Rnet_S_2,Hnet_2, criterion):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Rnet_C_1.eval()
    Rnet_S_1.eval()
    Hnet_1.eval()
    Rnet_C_2.eval()
    Rnet_S_2.eval()
    Hnet_2.eval()
    batch_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    RClosses = AverageMeter()
    SumLosses = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    RCdiff = AverageMeter()

    for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(val_loader, 0):

        '''cover_imgv, container_img, secret_imgv_nh, rev_secret_img, rev_cover_img, errH, errR, errRC, diffH, diffR, diffRC\
            = forward_pass_val(secret_img, secret_target, cover_img, cover_target, Rnet_C, Rnet_S, Hnet, criterion, val_cover=1)
        deta_secret = secret_img.cuda()
        batch_size_secret, channel_secret, h, w = secret_img.size()
        batch_size_cover, channel_cover, h_s, w_s = cover_img.size()
        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()
        deta_cover = cover_img.cuda()
        deta_container = torch.zeros(batch_size_cover,channel_cover,h,w).to(cover_img.device)
        sum_container = torch.zeros(batch_size_cover,channel_cover,h,w).to(cover_img.device)
        sum_cover = torch.zeros(batch_size_cover,channel_cover,h,w).to(cover_img.device)
        sum_secret = torch.zeros(batch_size_secret,channel_secret,h_s,w_s).to(cover_img.device)'''

        batch_size_secret, channel_secret, h, w = secret_img.size()
        batch_size_cover, channel_cover, h_s, w_s = cover_img.size()

        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()

        deta_secret = secret_img.view(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret,
                                      opt.imageSize,
                                      opt.imageSize).repeat(opt.num_training, 1, 1, 1)
        deta_cover = cover_img.view(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                                    opt.imageSize)
        deta_container = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                                     opt.imageSize).to(cover_img.device)

        sum_container = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                                    opt.imageSize).to(cover_img.device)
        last_container_img = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                                opt.imageSize).to(cover_img.device)
        sum_cover = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                                opt.imageSize).to(cover_img.device)
        sum_secret = torch.zeros(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret, opt.imageSize,
                                 opt.imageSize).repeat(opt.num_training, 1, 1, 1).to(cover_img.device)
        cover_imgv_1, container_img_1, secret_imgv_nh_1, rev_secret_img_1, rev_cover_img_1 \
            = forward_pass_1(deta_secret, secret_target, deta_cover, cover_target, Rnet_C_1, Rnet_S_1, Hnet_1,
                             deta_container)
        res_cover = (cover_img - rev_cover_img_1)
        res_secret = (secret_img - rev_secret_img_1)
        deta_cover_1 = res_cover.abs()
        deta_secret_1 = res_secret.abs()
        #deta_cover_1 = rev_cover_img_1
        #deta_secret_1 = torch.cat([deta_secret_1, rev_secret_img_1], 1)
        deta_container_1 = container_img_1
        cover_imgv_2, container_img_2, new_container_img_2, secret_imgv_nh_2, rev_secret_img_2, rev_cover_img_2 \
            = forward_pass_2(deta_secret_1, secret_target, deta_cover_1, cover_target, Rnet_C_2, Rnet_S_2, Hnet_2,
                             deta_container_1)
        errH = criterion(new_container_img_2, cover_img)  # Hiding net container
        errR = criterion(rev_secret_img_2, secret_img)  # Reveal net secret
        errRC = criterion(rev_cover_img_2, cover_img)  # Reveal net cover

        # L1 metric
        diffH = (new_container_img_2 - cover_img).abs().mean() * 255  ## Hiding net container
        diffR = (rev_secret_img_2 - secret_img).abs().mean() * 255  # Reveal net secret
        diffRC = (rev_cover_img_2 - cover_img).abs().mean() * 255  # Reveal net cover


        Hlosses.update(errH.item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        Rlosses.update(errR.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        RClosses.update(errRC.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        Hdiff.update(diffH.item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Rdiff.update(diffR.item(), opt.bs_secret * opt.num_secret * opt.num_training)
        RCdiff.update(diffRC.item(), opt.bs_secret * opt.num_secret * opt.num_training)


        '''if i == 0:
            save_result_pic(opt.bs_secret * opt.num_training, cover_imgv, container_img.data, secret_imgv_nh,
                            rev_secret_img.data, epoch, i, opt.validationpics)
        if epoch == opt.epochs and i % opt.resultPicFrequency == 0:
            save_result_pic(opt.bs_secret * opt.num_training, cover_imgv, container_img.data, secret_imgv_nh,
                            rev_secret_img.data, epoch, i, opt.trainpics)'''
        if opt.num_secret >= 6:
            i_total = 80
        else:
            i_total = 200
        if i == i_total - 1:
            break

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_RCloss = %.6f\t val_Hdiff = %.6f\t val_Rdiff=%.2f\t val_RCdiff=%.2f\t batch time=%.2f" % (
            epoch, Hlosses.val, Rlosses.val, RClosses.val, Hdiff.val, Rdiff.val, RCdiff.val, batch_time.val)
        if i % opt.logFrequency == 0:
            print(val_log)

    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_RCloss = %.6f\t val_Hdiff = %.4f\t val_Rdiff=%.4f\t val_RCdiff=%.2f\t validation time=%.2f" % (
        epoch, Hlosses.avg, Rlosses.avg, RClosses.avg, Hdiff.avg, Rdiff.avg, RCdiff.avg, batch_time.sum)
    print_log(val_log, logPath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
        writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
        writer.add_scalar('validation/RC_loss_avg', RClosses.avg, epoch)
        writer.add_scalar('validation/H_diff_avg', Hdiff.avg, epoch)
        writer.add_scalar('validation/R_diff_avg', Rdiff.avg, epoch)
        writer.add_scalar('validation/RC_diff_avg', RCdiff.avg, epoch)

    print(
        "#################################################### validation end ########################################################")
    return Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg


#def analysis(val_loader, epoch, Hnet, Rnet, HnetD, RnetD, criterion):
def analysis(val_loader, epoch, Rnet_C_1, Rnet_S_1, Hnet_1, Rnet_C_2, Rnet_S_2, Hnet_2, criterion):
    print(
        "#################################################### analysis begin ########################################################")

    Rnet_C_1.eval()
    Rnet_S_1.eval()
    Hnet_1.eval()
    Rnet_C_2.eval()
    Rnet_S_2.eval()
    Hnet_2.eval()

    #HnetD.eval()
    #RnetD.eval()
    import warnings
    warnings.filterwarnings("ignore")

    for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(val_loader, 0):

        ####################################### Cover Agnostic #######################################
        '''cover_imgv, container_img, secret_imgv_nh, rev_secret_img, rev_cover_img, errH, errR, errRC, diffH, diffR, diffRC \
            = forward_pass(secret_img, secret_target, cover_img, cover_target, Rnet_C, Rnet_S, Hnet, criterion, val_cover=1)
        secret_encoded = container_img - cover_imgv'''

        '''save_result_pic_analysis(opt.bs_secret * opt.num_training, cover_imgv.clone(), container_img.clone(),
                                 secret_imgv_nh.clone(), rev_secret_img.clone(), epoch, i, opt.validationpics)'''
        batch_size_secret, channel_secret, h, w = secret_img.size()
        batch_size_cover, channel_cover, h_s, w_s = cover_img.size()

        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()

        deta_secret = secret_img.view(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret,
                                      opt.imageSize,
                                      opt.imageSize).repeat(opt.num_training, 1, 1, 1)
        deta_cover = cover_img.view(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                                    opt.imageSize)
        deta_container = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                                     opt.imageSize).to(cover_img.device)

        sum_container = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                                    opt.imageSize).to(cover_img.device)
        last_container_img = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover,
                                         opt.imageSize,
                                         opt.imageSize).to(cover_img.device)
        sum_cover = torch.zeros(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                                opt.imageSize).to(cover_img.device)
        sum_secret = torch.zeros(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret, opt.imageSize,
                                 opt.imageSize).repeat(opt.num_training, 1, 1, 1).to(cover_img.device)
        cover_imgv_1, container_img_1, secret_imgv_nh_1, rev_secret_img_1, rev_cover_img_1 \
            = forward_pass_1(deta_secret, secret_target, deta_cover, cover_target, Rnet_C_1, Rnet_S_1, Hnet_1,
                             deta_container)
        res_cover = (cover_img - rev_cover_img_1)
        res_secret = (secret_img - rev_secret_img_1)
        deta_cover_1 = res_cover.abs()
        deta_secret_1 = res_secret.abs()
        # deta_cover_1 = rev_cover_img_1
        # deta_secret_1 = torch.cat([deta_secret_1, rev_secret_img_1], 1)
        deta_container_1 = container_img_1
        cover_imgv_2, container_img_2, new_container_img_2, secret_imgv_nh_2, rev_secret_img_2, rev_cover_img_2 \
            = forward_pass_2(deta_secret_1, secret_target, deta_cover_1, cover_target, Rnet_C_2, Rnet_S_2, Hnet_2,
                             deta_container_1)

        N, _, _, _ = rev_secret_img_2.shape

        cover_img_numpy = cover_imgv_1.clone().cpu().detach().numpy()
        #container_img_numpy = container_img.clone().cpu().detach().numpy()
        container_img_numpy = new_container_img_2.clone().cpu().detach().numpy()

        cover_img_numpy = cover_img_numpy.transpose(0, 2, 3, 1)
        container_img_numpy = container_img_numpy.transpose(0, 2, 3, 1)

        rev_secret_numpy = rev_secret_img_2.cpu().detach().numpy()
        secret_img_numpy = secret_imgv_nh_1.cpu().detach().numpy()

        rev_secret_numpy = rev_secret_numpy.transpose(0, 2, 3, 1)
        secret_img_numpy = secret_img_numpy.transpose(0, 2, 3, 1)
        diffH = (new_container_img_2 - cover_img).abs().mean() * 255  ## Hiding net container
        diffR = (rev_secret_img_2 - secret_img).abs().mean() * 255  # Reveal net secret
        diffRC = (rev_cover_img_2 - cover_img).abs().mean() * 255  # Reveal net cover

        # PSNR
        print("Cover Agnostic")

        print("Secret APD C:", diffH.item())

        psnr = np.zeros((N, 3))
        for i in range(N):
            psnr[i, 0] = PSNR(cover_img_numpy[i, :, :, 0], container_img_numpy[i, :, :, 0])
            psnr[i, 1] = PSNR(cover_img_numpy[i, :, :, 1], container_img_numpy[i, :, :, 1])
            psnr[i, 2] = PSNR(cover_img_numpy[i, :, :, 2], container_img_numpy[i, :, :, 2])
        print("Avg. PSNR C:", psnr.mean().item())

        # SSIM
        ssim = np.zeros(N)
        for i in range(N):
            ssim[i] = SSIM(cover_img_numpy[i], container_img_numpy[i], multichannel=True)
        print("Avg. SSIM C:", ssim.mean().item())

        # LPIPS
        '''import PerceptualSimilarity.models
        model = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        lpips = model.forward(cover_imgv, container_img)
        print("Avg. LPIPS C:", lpips.mean().item())'''

        print("Secret APD S:", diffR.item())

        psnr = np.zeros(N)
        for i in range(N):
            psnr[i] = PSNR(secret_img_numpy[i], rev_secret_numpy[i])
        print("Avg. PSNR S:", psnr.mean().item())

        # SSIM
        ssim = np.zeros(N)
        for i in range(N):
            ssim[i] = SSIM(secret_img_numpy[i], rev_secret_numpy[i], multichannel=True)
        print("Avg. SSIM S:", ssim.mean().item())

        # LPIPS
        '''import PerceptualSimilarity.models
        model = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        lpips = model.forward(secret_imgv_nh, rev_secret_img)
        print("Avg. LPIPS S:", lpips.mean().item())'''

        print("*******DONE!**********")

        break


def analysis_origion(val_loader, epoch, Rnet_C, Rnet_S, Hnet, criterion):
    print(
        "#################################################### analysis begin ########################################################")

    Rnet_C.eval()
    Rnet_S.eval()
    Hnet.eval()

    #HnetD.eval()
    #RnetD.eval()
    import warnings
    warnings.filterwarnings("ignore")

    for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(val_loader, 0):

        ####################################### Cover Agnostic #######################################
        '''cover_imgv, container_img, secret_imgv_nh, rev_secret_img, errH, errR, diffH, diffR \
            = forward_pass(secret_img, secret_target, cover_img, cover_target, Hnet_C, Hnet_S, Rnet, criterion, val_cover=1)'''
        noise_types = ['identity', 'dropout', 'gaussian', 'jpeg_50', 'jpeg_85']
        for noise_type in noise_types:
            print(noise_type)
            cover_imgv, container_img, secret_imgv_nh, rev_secret_img, rev_cover_img, errH, errR, errRC, diffH, diffR, diffRC \
                = forward_pass_analysis(secret_img, secret_target, cover_img, cover_target, Rnet_C, Rnet_S, Hnet,
                                        criterion, val_cover=1, noise_type=noise_type)
            secret_encoded = container_img - cover_imgv

            '''save_result_pic_analysis(opt.bs_secret * opt.num_training, cover_imgv.clone(), container_img.clone(),
                                 secret_imgv_nh.clone(), rev_secret_img.clone(), epoch, i, opt.validationpics)'''

            N, _, _, _ = rev_secret_img.shape

            cover_img_numpy = cover_imgv.clone().cpu().detach().numpy()
            container_img_numpy = container_img.clone().cpu().detach().numpy()

            cover_img_numpy = cover_img_numpy.transpose(0, 2, 3, 1)
            container_img_numpy = container_img_numpy.transpose(0, 2, 3, 1)

            rev_secret_numpy = rev_secret_img.cpu().detach().numpy()
            secret_img_numpy = secret_imgv_nh.cpu().detach().numpy()

            rev_secret_numpy = rev_secret_numpy.transpose(0, 2, 3, 1)
            secret_img_numpy = secret_img_numpy.transpose(0, 2, 3, 1)

            # PSNR
            print("Cover Agnostic")

            print("Secret APD C:", diffH.item())

            psnr = np.zeros((N, 3))
            for i in range(N):
                psnr[i, 0] = PSNR(cover_img_numpy[i, :, :, 0], container_img_numpy[i, :, :, 0])
                psnr[i, 1] = PSNR(cover_img_numpy[i, :, :, 1], container_img_numpy[i, :, :, 1])
                psnr[i, 2] = PSNR(cover_img_numpy[i, :, :, 2], container_img_numpy[i, :, :, 2])
            print("Avg. PSNR C:", psnr.mean().item())

            # SSIM
            ssim = np.zeros(N)
            for i in range(N):
                ssim[i] = SSIM(cover_img_numpy[i], container_img_numpy[i], multichannel=True)
            print("Avg. SSIM C:", ssim.mean().item())

            # LPIPS
            import PerceptualSimilarity.models
            model = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
            lpips = model.forward(cover_imgv, container_img)
            print("Avg. LPIPS C:", lpips.mean().item())

            print("Secret APD S:", diffR.item())

            psnr = np.zeros(N)
            for i in range(N):
                psnr[i] = PSNR(secret_img_numpy[i], rev_secret_numpy[i])
            print("Avg. PSNR S:", psnr.mean().item())

            # SSIM
            ssim = np.zeros(N)
            for i in range(N):
                ssim[i] = SSIM(secret_img_numpy[i], rev_secret_numpy[i], multichannel=True)
            print("Avg. SSIM S:", ssim.mean().item())

            # LPIPS
            import PerceptualSimilarity.models
            model = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
            lpips = model.forward(secret_imgv_nh, rev_secret_img)
            print("Avg. LPIPS S:", lpips.mean().item())

            print("*******DONE!**********")

        break


def forward_pass_analysis(secret_img, secret_target, cover_img, cover_target, Rnet_C, Rnet_S, Hnet, criterion, val_cover=0,
                          i_c=None, position=None, Se_two=None, noise_type='identity'):
    batch_size_secret, channel_secret, _, _ = secret_img.size()
    batch_size_cover, channel_cover, _, _ = cover_img.size()

    if opt.cuda:
        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()
        # concat_img = concat_img.cuda()

    secret_imgv = secret_img.view(batch_size_secret // opt.num_secret, channel_secret * opt.num_secret, opt.imageSize,
                                  opt.imageSize)
    secret_imgv_nh = secret_imgv.repeat(opt.num_training, 1, 1, 1)

    cover_img = cover_img.view(batch_size_cover // opt.num_cover, channel_cover * opt.num_cover, opt.imageSize,
                               opt.imageSize)

    if opt.no_cover and (
            val_cover == 0):  # if val_cover = 1, always use cover in val; otherwise, no_cover True >>> not using cover in training
        cover_img.fill_(0.0)
    if (opt.plain_cover or opt.noise_cover) and (val_cover == 0):
        cover_img.fill_(0.0)
    b, c, w, h = cover_img.size()
    if opt.plain_cover and (val_cover == 0):
        img_w1 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_w2 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_w3 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_w4 = torch.cat((torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda(),
                            torch.rand(b, c, 1, 1).repeat(1, 1, w // 4, h // 4).cuda()), dim=2)
        img_wh = torch.cat((img_w1, img_w2, img_w3, img_w4), dim=3)
        cover_img = cover_img + img_wh
    if opt.noise_cover and (val_cover == 0):
        cover_img = cover_img + ((torch.rand(b, c, w, h) - 0.5) * 2 * 0 / 255).cuda()

    cover_imgv = cover_img

    if opt.cover_dependent:
        H_input = torch.cat((cover_imgv, secret_imgv), dim=1)
    else:
        H_input = secret_imgv

    itm_secret_img = Hnet(H_input)
    #out_dct_1, out_dct_2, out_dct_3, out_dct_4, itm_secret_img = Hnet_S(H_input)
    if i_c != None:
        if type(i_c) == type(1.0):
            #######To keep one channel
            itm_secret_img_clone = itm_secret_img.clone()
            itm_secret_img.fill_(0)
            itm_secret_img[:, int(i_c):int(i_c) + 1, :, :] = itm_secret_img_clone[:, int(i_c):int(i_c) + 1, :, :]
        if type(i_c) == type(1):
            print('aaaaa', i_c)
            #######To set one channel to zero
            itm_secret_img[:, i_c:i_c + 1, :, :].fill_(0.0)

    if position != None:
        itm_secret_img[:, :, position:position + 1, position:position + 1].fill_(0.0)
    if Se_two == 2:
        itm_secret_img_half = itm_secret_img[0:batch_size_secret // 2, :, :, :]
        itm_secret_img = itm_secret_img + torch.cat((itm_secret_img_half.clone().fill_(0.0), itm_secret_img_half), 0)
    elif type(Se_two) == type(0.1):
        itm_secret_img = itm_secret_img + Se_two * torch.rand(itm_secret_img.size()).cuda()
    if opt.cover_dependent:
        container_img = itm_secret_img
    else:
        itm_secret_img = itm_secret_img.repeat(opt.num_training, 1, 1, 1)
        container_img = itm_secret_img + cover_imgv
        #container_img = Hnet_C(cover_img, out_dct_1, out_dct_2, out_dct_3, out_dct_4, itm_secret_img)


    errH = criterion(container_img, cover_imgv)  # Hiding net

    if not os.path.exists("jpgs_50"):
        os.makedirs("jpgs_50")
    if not os.path.exists("jpgs_85"):
        os.makedirs("jpgs_85")

    if noise_type == 'identity':
        container_img_noised = noiser_identity([container_img, cover_img])[0]
    elif noise_type == 'jpeg_50':
        container_img_copy = container_img.clone()
        containers_ori = container_img_copy.detach().cpu().numpy()

        containers = np.transpose(containers_ori, (0, 2, 3, 1))
        N, _, _, _ = containers.shape
        containers = (np.clip(containers, 0.0, 1.0) * 255).astype(np.uint8)
        for i in range(N):
            img = cv2.cvtColor(containers[i], cv2.COLOR_RGB2BGR)
            folder_imgs = "jpgs_" + str(50) + "/jpg_" + str(i).zfill(2) + ".jpg"
            cv2.imwrite(folder_imgs, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            # cv2.imwrite("jpgs/jpg_" + str(i).zfill(2) + ".png", img)

        containers_loaded = np.copy(containers)
        for i in range(N):
            folder_imgs = "jpgs_" + str(50) + "/jpg_" + str(i).zfill(2) + ".jpg"
            img = cv2.imread(folder_imgs)
            # import pdb; pdb.set_trace()
            # img = cv2.imread("jpgs/jpg_" + str(i).zfill(2) + ".png")
            containers_loaded[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        containers_loaded = np.transpose(containers_loaded, (0, 3, 1, 2)).astype(np.float32) / 255

        container_gap = containers_loaded - containers_ori
        container_gap = torch.from_numpy(container_gap).float().cuda()
        # import pdb; pdb.set_trace()

        container_img_noised = container_img + container_gap
    elif noise_type == 'jpeg_85':
        container_img_copy = container_img.clone()
        containers_ori = container_img_copy.detach().cpu().numpy()

        containers = np.transpose(containers_ori, (0, 2, 3, 1))
        N, _, _, _ = containers.shape
        containers = (np.clip(containers, 0.0, 1.0) * 255).astype(np.uint8)
        for i in range(N):
            img = cv2.cvtColor(containers[i], cv2.COLOR_RGB2BGR)
            folder_imgs = "jpgs_" + str(85) + "/jpg_" + str(i).zfill(2) + ".jpg"
            cv2.imwrite(folder_imgs, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            # cv2.imwrite("jpgs/jpg_" + str(i).zfill(2) + ".png", img)

        containers_loaded = np.copy(containers)
        for i in range(N):
            folder_imgs = "jpgs_" + str(85) + "/jpg_" + str(i).zfill(2) + ".jpg"
            img = cv2.imread(folder_imgs)
            # import pdb; pdb.set_trace()
            # img = cv2.imread("jpgs/jpg_" + str(i).zfill(2) + ".png")
            containers_loaded[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        containers_loaded = np.transpose(containers_loaded, (0, 3, 1, 2)).astype(np.float32) / 255

        container_gap = containers_loaded - containers_ori
        container_gap = torch.from_numpy(container_gap).float().cuda()
        # import pdb; pdb.set_trace()

        container_img_noised = container_img + container_gap
    elif noise_type == 'dropout':
        container_img_noised = noiser_dropout([container_img, cover_img])[0]
    elif noise_type == 'gaussian':
        container_img_noised = noiser_gaussian(container_img)
    elif noise_type == 'cropout':
        container_img_noised = noiser_cropout([container_img, cover_img])[0]
    else:
        container_img_noised = noiser_crop([container_img, cover_img])[0]

    rev_secret_img = Rnet_S(container_img_noised)
    rev_cover_img = Rnet_C(container_img_noised)

    if noise_type == 'crop':
        errR = criterion(rev_secret_img, noiser_crop([secret_imgv_nh, cover_img])[0])
        errRC = criterion(rev_cover_img, noiser_crop([container_img, cover_img])[0])
    else:
        errR = criterion(rev_secret_img, secret_imgv_nh)  # Reveal net
        errRC = criterion(rev_cover_img, cover_imgv)  # Reveal net

    # L1 metric
    diffH = (container_img - cover_imgv).abs().mean() * 255

    if noise_type == 'crop':
        diffR = (rev_secret_img - noiser_crop([secret_imgv_nh, cover_img])[0]).abs().mean() * 255
        diffRC = (rev_cover_img - noiser_crop([container_img, cover_img])[0]).abs().mean() * 255
    elif noise_type == 'cropout':
        h_start, h_end, w_start, w_end = noiser_cropout.get_crop_coords()
        diffR = (rev_secret_img[:, :, h_start:h_end, w_start:w_end] - secret_imgv_nh[:, :, h_start:h_end,
                                                                      w_start:w_end]).abs().mean() * 255
        diffRC = (rev_cover_img[:, :, h_start:h_end, w_start:w_end] - cover_imgv[:, :, h_start:h_end,
                                                                      w_start:w_end]).abs().mean() * 255
    else:
        diffR = (rev_secret_img - secret_imgv_nh).abs().mean() * 255
        diffRC = (rev_cover_img - cover_imgv).abs().mean() * 255
    return cover_imgv, container_img, secret_imgv_nh, rev_secret_img, rev_cover_img, errH, errR, errRC, diffH, diffR, diffRC


def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not opt.debug:
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic_analysis(bs_secret_times_num_training, cover, container, secret, rev_secret, epoch, i,
                             save_path=None, postname=''):
    path = './qualitative_results/'
    if not os.path.exists(path):
        os.makedirs(path)
    resultImgName = path + 'universal_qualitative_results.png'

    cover = cover[:4]
    container = container[:4]
    secret = secret[:4]
    rev_secret = rev_secret[:4]

    cover_gap = container - cover
    secret_gap = rev_secret - secret
    cover_gap = (cover_gap * 10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap * 10 + 0.5).clamp_(0.0, 1.0)

    for i_cover in range(4):
        cover_i = cover[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]
        container_i = container[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]
        cover_gap_i = cover_gap[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]

        if i_cover == 0:
            showCover = torch.cat((cover_i, container_i, cover_gap_i), 0)
        else:
            showCover = torch.cat((showCover, cover_i, container_i, cover_gap_i), 0)

    for i_secret in range(4):
        secret_i = secret[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]
        rev_secret_i = rev_secret[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]
        secret_gap_i = secret_gap[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]

        if i_secret == 0:
            showSecret = torch.cat((secret_i, rev_secret_i, secret_gap_i), 0)
        else:
            showSecret = torch.cat((showSecret, secret_i, rev_secret_i, secret_gap_i), 0)

    showAll = torch.cat((showCover, showSecret), 0)
    showAll = showAll.reshape(6, 4, 3, 128, 128)
    showAll = showAll.permute(1, 0, 2, 3, 4)
    showAll = showAll.reshape(4 * 6, 3, 128, 128)
    vutils.save_image(showAll, resultImgName, nrow=6, padding=1, normalize=False)


# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic(bs_secret_times_num_training, cover, container, secret, rev_secret, epoch, i, save_path=None,
                    postname=''):
    # if not opt.debug:
    # cover=container: bs*nt/nc;   secret=rev_secret: bs*nt/3*nh
    if opt.debug:
        save_path = './debug/debug_images'
    resultImgName = '%s/ResultPics_epoch%03d_batch%04d%s.png' % (save_path, epoch, i, postname)

    cover_gap = container - cover
    secret_gap = rev_secret - secret
    cover_gap = (cover_gap * 10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap * 10 + 0.5).clamp_(0.0, 1.0)
    # print(cover_gap.abs().sum(dim=-1).sum(dim=-1).sum(dim=-1), secret_gap.abs().sum(dim=-1).sum(dim=-1).sum(dim=-1))

    # showCover = torch.cat((cover, container, cover_gap),0)

    for i_cover in range(opt.num_cover):
        cover_i = cover[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]
        container_i = container[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]
        cover_gap_i = cover_gap[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]

        if i_cover == 0:
            showCover = torch.cat((cover_i, container_i, cover_gap_i), 0)
        else:
            showCover = torch.cat((showCover, cover_i, container_i, cover_gap_i), 0)

    for i_secret in range(opt.num_secret):
        secret_i = secret[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]
        rev_secret_i = rev_secret[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]
        secret_gap_i = secret_gap[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]

        if i_secret == 0:
            showSecret = torch.cat((secret_i, rev_secret_i, secret_gap_i), 0)
        else:
            showSecret = torch.cat((showSecret, secret_i, rev_secret_i, secret_gap_i), 0)

    if opt.channel_secret == opt.channel_cover:
        showAll = torch.cat((showCover, showSecret), 0)
        vutils.save_image(showAll, resultImgName, nrow=bs_secret_times_num_training, padding=1, normalize=True)
    else:
        ContainerImgName = '%s/ContainerPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        SecretImgName = '%s/SecretPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(showCover, ContainerImgName, nrow=bs_secret_times_num_training, padding=1, normalize=True)
        vutils.save_image(showSecret, SecretImgName, nrow=bs_secret_times_num_training, padding=1, normalize=True)

def Normalization_01(res_cover):
    res_cover_max = torch.max(res_cover)
    res_cover_min = torch.min(res_cover)
    new_cover = (res_cover-res_cover_min)/(res_cover_max-res_cover_min)
    return new_cover

def gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, padding=1, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()