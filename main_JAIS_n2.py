# encoding: utf-8

import argparse
import os
import shutil
import socket
import time
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# import utils.transformed as transforms
from torchvision import transforms
# from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet_3_coarse import UnetGenerator_coarse
from models.HidingUNet_3_fine import UnetGenerator_fine
from models.RevealNet_C import RevealNet_C
from models.RevealNet_S import RevealNet_S
from models.RevealUNet_3 import UnetReveal_3
from torchvision.datasets import ImageFolder
import pdb
import math
import random
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
#from skimage.measure import compare_ssim as SSIM, compare_psnr as PSNR


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
parser.add_argument('--Hnet_C', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Hnet_S', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
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
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--alpha', type=float, default=0.75,
                    help='hyper parameter of alpha')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='checkpoint folder')
parser.add_argument('--test_diff', default='', help='another checkpoint folder')
parser.add_argument('--checkpoint', default='', help='checkpoint address')
parser.add_argument('--checkpoint_diff', default='', help='another checkpoint address')

parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=100, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')
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
    global opt, optimizer, optimizerR, writer, logPath, scheduler, schedulerR, val_loader, smallestLoss, DATA_DIR

    opt = parser.parse_args()
    #opt.ngpu = torch.cuda.device_count()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

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
                experiment_dir = opt.remark + "_" + "beta" + str(opt.beta) + "_" +"num_secret"+str(opt.num_secret) +"num_cover"+str(opt.num_cover)+ "_" + cur_time
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
    traindir = os.path.join('/data/data/Imagenet2012/ILSVRC2012_img_train')
    valdir = os.path.join('/data/data/Imagenet2012/ILSVRC2012_img_val')
    # coverdir = os.path.join('/data/zhangle/DAH_v/dataset/C1S1/num_5/cover')
    #
    # secretdir = os.path.join('/data/zhangle/DAH_v/dataset/C1S1/num_5/secret')
    coverdir = os.path.join('/data/zhangle/UDH_v/dataset/val_30/cover')
    secretdir = os.path.join('/data/zhangle/UDH_v/dataset/val_30/secret')
    print(coverdir)
    #coverdir = os.path.join('/data/zhangle/DAH_v/dataset/stegexpose/cover')
    #secretdir = os.path.join('/data/zhangle/DAH_v/dataset/stegexpose/secret')

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
        test_v_dataset_cover = ImageFolder(
            coverdir,
            transforms_cover)
        test_v_dataset_secret = ImageFolder(
            secretdir,
            transforms_secret)
        assert test_dataset_cover;
        assert test_dataset_secret
        assert test_v_dataset_cover;
        assert test_v_dataset_secret

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
        Hnet_C = UnetGenerator_coarse(input_nc=opt.channel_secret * opt.num_secret + opt.channel_cover * opt.num_cover,
                             output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs, norm_layer=norm_layer,
                             use_tanh=0)
        Hnet_F = UnetGenerator_fine(input_nc=opt.channel_secret * opt.num_secret + opt.channel_cover * opt.num_cover,
                                      output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs,
                                      norm_layer=norm_layer,
                                      use_tanh=0)
        '''Hnet = UnetGenerator(input_nc=opt.channel_secret * opt.num_secret + opt.channel_cover * opt.num_cover,
                             output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs, norm_layer=norm_layer,
                             output_function=nn.Sigmoid)
        Hnet_C = UnetGenerator_coarse(input_nc=opt.channel_secret * opt.num_cover,
                                      output_nc=opt.channel_cover * opt.num_cover,
                                      num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Tanh)
        Hnet_F = UnetGenerator_fine(input_nc=opt.channel_secret * opt.num_secret,
                                    output_nc=opt.channel_cover * opt.num_cover,
                                    num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Tanh)'''
    else:
        #print('opt.beta',opt.beta)
        Hnet_C = UnetGenerator_coarse(input_nc=opt.channel_secret * opt.num_secret,
                             output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs, norm_layer=norm_layer,
                             use_tanh=1)
        Hnet_F = UnetGenerator_fine(input_nc=opt.channel_secret * opt.num_secret,
                             output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs, norm_layer=norm_layer,
                             use_tanh=1)

    Rnet_C = RevealNet_C(input_nc=opt.channel_cover * opt.num_cover, output_nc=opt.channel_cover * opt.num_cover, nhf=64,
                     norm_layer=norm_layer, output_function=nn.Sigmoid)
    Rnet_S = RevealNet_S(input_nc=opt.channel_cover * opt.num_cover, output_nc=opt.channel_secret * opt.num_secret, nhf=64,
                     norm_layer=norm_layer, output_function=nn.Sigmoid)
    RUnet_3 = UnetReveal_3(input_nc=opt.channel_cover * opt.num_cover, output_nc=opt.channel_secret * opt.num_secret, num_downs=num_downs,
                                      norm_layer=norm_layer,
                                      use_tanh=0)


    if opt.cover_dependent:
        assert opt.num_training == 1
        assert opt.no_cover == False

    ##### We used kaiming normalization #####


    ##### Always set to multiple GPU mode  #####
    Hnet_C = torch.nn.DataParallel(Hnet_C).cuda()
    Hnet_F = torch.nn.DataParallel(Hnet_F).cuda()
    Rnet_C = torch.nn.DataParallel(Rnet_C).cuda()
    Rnet_S = torch.nn.DataParallel(Rnet_S).cuda()
    RUnet_3 = torch.nn.DataParallel(RUnet_3).cuda()




    if opt.checkpoint != "":
        checkpoint = torch.load(opt.checkpoint)
        Hnet_C.load_state_dict(checkpoint['H_C_state_dict'])
        Hnet_F.load_state_dict(checkpoint['H_F_state_dict'])
        Rnet_C.load_state_dict(checkpoint['R_C_state_dict'])
        Rnet_S.load_state_dict(checkpoint['R_S_state_dict'])
        RUnet_3.load_state_dict(checkpoint['R_3_state_dict'])


    # Loss and Metric
    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    # Train the networks when opt.test is empty
    if opt.test == '':
        if not opt.debug:
            writer = SummaryWriter(log_dir='runs/' + experiment_dir)
        params = list(Hnet_C.parameters()) + list(Rnet_C.parameters())+ list(Rnet_S.parameters()) + list(RUnet_3.parameters())+ list(Hnet_F.parameters())
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
            train(train_loader, epoch, Hnet_C=Hnet_C, Hnet_F=Hnet_F, Rnet_C=Rnet_C, Rnet_S=Rnet_S, RUnet_3=RUnet_3, criterion=criterion)

            ####################### validation  #####################################
            val_hloss, val_rloss, val_hdiff, val_rdiff = validation(val_loader, epoch, Hnet_C=Hnet_C, Hnet_F=Hnet_F, Rnet_C=Rnet_C, Rnet_S=Rnet_S, RUnet_3=RUnet_3,
                                                                    criterion=criterion)

            ####################### adjust learning rate ############################
            scheduler.step(val_rloss)

            # Save the best model parameters
            sum_diff = val_hdiff + val_rdiff
            is_best = sum_diff < globals()["smallestLoss"]
            globals()["smallestLoss"] = sum_diff

            save_checkpoint({
                'epoch': epoch + 1,
                'H_C_state_dict': Hnet_C.state_dict(),
                'H_F_state_dict': Hnet_F.state_dict(),
                'R_C_state_dict': Rnet_C.state_dict(),
                'R_S_state_dict': Rnet_S.state_dict(),
                'R_3_state_dict': RUnet_3.state_dict(),
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
        test_v_loader_secret = DataLoader(test_v_dataset_secret, batch_size=opt.bs_secret * opt.num_secret,
                                          shuffle=True, num_workers=int(opt.workers))
        test_v_loader_cover = DataLoader(test_v_dataset_cover,
                                         batch_size=opt.bs_secret * opt.num_cover * opt.num_training,
                                         shuffle=True, num_workers=int(opt.workers))
        test_v_loader = zip(test_v_loader_secret, test_v_loader_cover)

        # analysis(test_loader, 0, Hnet_C=Hnet_C, Hnet_F=Hnet_F, Rnet_C=Rnet_C, Rnet_S=Rnet_S, RUnet_3=RUnet_3, criterion=criterion)
        analysis_v(test_loader, 0, Hnet_C=Hnet_C, Hnet_F=Hnet_F, Rnet_C=Rnet_C, Rnet_S=Rnet_S, RUnet_3=RUnet_3,
                 criterion=criterion)


def analysis_v(val_loader, epoch, Hnet_C, Hnet_S, Rnet, criterion):
    print(
        "#################################################### analysis begin ########################################################")

    # Hnet_C.eval()
    # Hnet_S.eval()
    # Rnet.eval()
    Hnet_C.eval()
    Hnet_F.eval()
    Rnet_C.eval()
    Rnet_S.eval()
    RUnet_3.eval()


    import warnings
    warnings.filterwarnings("ignore")


    for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(val_loader, 0):
        #vutils.save_image(cover_img, 'cover_img.png', nrow=1, padding=0, normalize=True)

        ####################################### Cover Agnostic #######################################
        # cover_imgv, container_img, secret_imgv_nh, rev_secret_img, errH, errR, diffH, diffR \
        #     = forward_pass(secret_img, secret_target, cover_img, cover_target, Hnet_C, Hnet_S, Rnet, criterion, val_cover=1)
        cover_imgv, final_container, secret_imgv_nh, final_rev_secret, errH, errH_3, errH_2, errR, errR_3, errR_2, diffH, diffH_3, diffH_2, diffR, diffR_3, diffR_2 \
            = forward_pass(secret_img, secret_target, cover_img, cover_target, Hnet_C, Hnet_F, Rnet_C, Rnet_S, RUnet_3,
                           criterion)
        container_img = final_container
        rev_secret_img = final_rev_secret
        secret_encoded = container_img - cover_imgv

        #save_result_pic_analysis_v(opt.bs_secret * opt.num_training, cover_imgv.clone(), container_img.clone(),secret_imgv_nh.clone(), rev_secret_img.clone(), epoch, i, opt.validationpics)
        #save_result_pic_v(opt.bs_secret * opt.num_training, cover_imgv, container_img.data, secret_imgv_nh, rev_secret_img.data, epoch, i, opt.validationpics)
        save_result_pic_v_2(opt.bs_secret * opt.num_training, cover_imgv, container_img.data, secret_imgv_nh,
                          rev_secret_img.data, epoch, i, opt.validationpics)
        print('************************DONE!****************************************')


        break
def save_result_pic_v_2(bs_secret_times_num_training, cover, container, secret, rev_secret, epoch, i, save_path=None,
                    postname=''):
    # if not opt.debug:
    # cover=container: bs*nt/nc;   secret=rev_secret: bs*nt/3*nh
    print('###############',cover.shape)
    print('%%%%%%%%%%%%%%%%%%',bs_secret_times_num_training)
    if opt.debug:
        save_path = './debug/debug_images'
    resultImgName = '%s/visulization_secret_num%03d_cover_num%04d%s.png' % (save_path, opt.num_secret, opt.num_cover, postname)

    cover_gap = container - cover
    secret_gap = rev_secret - secret
    cover_gap_10 = (cover_gap * 10 + 0.5).clamp_(0.0, 1.0)
    secret_gap_10 = (secret_gap * 10 + 0.5).clamp_(0.0, 1.0)
    cover_gap_5 = (cover_gap * 5 + 0.5).clamp_(0.0, 1.0)
    secret_gap_5 = (secret_gap * 5 + 0.5).clamp_(0.0, 1.0)
    cover_gap_1 = (cover_gap * 1 + 0.5).clamp_(0.0, 1.0)
    secret_gap_1 = (secret_gap * 1 + 0.5).clamp_(0.0, 1.0)
    '''print('cover_gap', cover_gap)
    print('secret_gap', secret_gap)
    print('cover_gap_1', cover_gap_1)
    print('secret_gap_1', secret_gap_1)
    print('cover_gap_5', cover_gap_5)
    print('secret_gap_5', secret_gap_5)
    print('cover_gap_10', cover_gap_10)
    print('secret_gap_10', secret_gap_10)'''
    # print(cover_gap.abs().sum(dim=-1).sum(dim=-1).sum(dim=-1), secret_gap.abs().sum(dim=-1).sum(dim=-1).sum(dim=-1))

    # showCover = torch.cat((cover, container, cover_gap),0)
    for i_cover in range(bs_secret_times_num_training):
        cover_i = cover[i_cover, :, :, :]
        container_i = container[i_cover, :, :, :]

        vutils.save_image(cover_i, '/data/zhangle/second/exampleresult/cover/cover_val_1_'+'_'+str(i_cover)+'.png', nrow=1, padding=0, normalize=True)
        vutils.save_image(container_i, '/data/zhangle/second/exampleresult/container/container_val_1_'+'_'+str(i_cover)+'.png', nrow=1, padding=0, normalize=True)
        # vutils.save_image(cover_gap_10_i, '/home/daisy/zhangle/UDH_v/stegexpose/DAH/cover_gap'+str(i_cover)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
    for i_secret in range(bs_secret_times_num_training):
        secret_i = secret[i_secret, :, :, :]

        # vutils.save_image(secret_i, '/data/zhangle/UDH_v/exampleresult/secret/secret_val_1_'+'_'+str(i_secret)+'.png', nrow=1, padding=0, normalize=True)


    for i_cover in range(opt.num_cover):
        cover_i = cover[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]
        container_i = container[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]
        cover_gap_10_i = cover_gap_10[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]
        cover_gap_5_i = cover_gap_5[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]
        cover_gap_1_i = cover_gap_1[:, i_cover * opt.channel_cover:(i_cover + 1) * opt.channel_cover, :, :]
        #vutils.save_image(cover_i, 'cover'+str(i_cover)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        #vutils.save_image(container_i, 'container'+str(i_cover)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        # vutils.save_image(cover_gap_10_i, 'cover_gap10_'+str(i_cover)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        # vutils.save_image(cover_gap_5_i, 'cover_gap5_' + str(i_cover) + '.png', nrow=bs_secret_times_num_training,
        #                   padding=0, normalize=True)
        # vutils.save_image(cover_gap_1_i, 'cover_gap1_' + str(i_cover) + '.png', nrow=bs_secret_times_num_training,
        #                   padding=0, normalize=True)


        if i_cover == 0:
            showCover = torch.cat((cover_i, container_i, cover_gap_10_i),
                                  0)
        else:
            showCover = torch.cat(
                (showCover, cover_i, container_i, cover_gap_10_i), 0)

    for i_secret in range(opt.num_secret):
        secret_i = secret[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]
        rev_secret_i = rev_secret[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]
        secret_gap_10_i = secret_gap_10[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]
        secret_gap_5_i = secret_gap_5[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]
        secret_gap_1_i = secret_gap_1[:, i_secret * opt.channel_secret:(i_secret + 1) * opt.channel_secret, :, :]
        #vutils.save_image(secret_i, 'secret'+str(i_secret)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        #vutils.save_image(rev_secret_i, 'rev_secret'+str(i_secret)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        # vutils.save_image(secret_gap_10_i, 'secret_gap10_'+str(i_secret)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        # vutils.save_image(secret_gap_5_i, 'secret_gap5_' + str(i_secret) + '.png', nrow=bs_secret_times_num_training,
        #                   padding=0, normalize=True)
        # vutils.save_image(secret_gap_1_i, 'secret_gap1_' + str(i_secret) + '.png', nrow=bs_secret_times_num_training,
        #                   padding=0, normalize=True)

        if i_secret == 0:
            showSecret = torch.cat(
                (secret_i, rev_secret_i, secret_gap_10_i), 0)
        else:
            showSecret = torch.cat(
                (showSecret, secret_i, rev_secret_i, secret_gap_10_i),
                0)

    if opt.channel_secret == opt.channel_cover:
        showAll = torch.cat((showCover, showSecret), 0)
        '''showAll = showAll.reshape(6, 4, 3, 128, 128)
        showAll = showAll.permute(1, 0, 2, 3, 4)
        showAll = showAll.reshape(4 * 6, 3, 128, 128)'''
        #cover = cover.reshape(1, 3, 128, 128)
        #vutils.save_image(showAll, resultImgName, nrow=bs_secret_times_num_training, padding=1, normalize=True)
        #vutils.save_image(cover, '/home/daisy/zhangle/UDH_v/stegexpose/DAH/cover'+str(epoch)+'_'+str(i)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        #vutils.save_image(container, '/home/daisy/zhangle/UDH_v/stegexpose/DAH/container'+str(epoch)+'_'+str(i)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        #vutils.save_image(cover_gap, 'cover_gap.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        #vutils.save_image(secret, '/home/daisy/zhangle/UDH_v/stegexpose/DAH/secret'+str(epoch)+'_'+str(i)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        #vutils.save_image(rev_secret, 'rev_secret.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        #vutils.save_image(secret_gap, 'secret_gap.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
    else:
        ContainerImgName = '%s/ContainerPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        SecretImgName = '%s/SecretPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(showCover, ContainerImgName, nrow=bs_secret_times_num_training, padding=1, normalize=True)
        vutils.save_image(showSecret, SecretImgName, nrow=bs_secret_times_num_training, padding=1, normalize=True)
def save_checkpoint(state, is_best, epoch, prefix):
    filename = '%s/checkpoint.pth.tar' % opt.outckpts

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/best_checkpoint.pth.tar' % opt.outckpts)
    if epoch == opt.epochs - 1:
        with open(opt.outckpts + prefix + '.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            # writer.writerow([epoch, loss, train1, train5, prec1, prec5])


def forward_pass(secret_img, secret_target, cover_img, cover_target, Hnet_C, Hnet_F, Rnet_C, Rnet_S, RUnet_3, criterion, val_cover=0, i_c=None,
                 position=None, Se_two=None):
    batch_size_secret, channel_secret, _, _ = secret_img.size()
    batch_size_cover, channel_cover, _, _ = cover_img.size()

    # Put tensors in GPU
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
    #print('cover_imgv.shape', cover_imgv.shape)
    #print('cover_imgv', cover_imgv)
    #print('cover_img.shape', cover_img.shape)
    #print('cover_img', cover_img)
    #++++++++++++++++++++++++++
    #print('secret_imgv.shape', secret_imgv.shape)
    #print('secret_imgv', secret_imgv)

    if opt.cover_dependent:
        input_1 = torch.cat((cover_imgv, secret_imgv), dim=1)

        cover_2= F.interpolate(cover_imgv, scale_factor=0.5)
        cover_3 = F.interpolate(cover_2, scale_factor=0.5)

        secret_2 = F.interpolate(secret_imgv, scale_factor=0.5)
        secret_3 = F.interpolate(secret_2, scale_factor=0.5)

        input_2 = torch.cat((cover_2, secret_2), dim=1)
        input_3 = torch.cat((cover_3, secret_3), dim=1)
    else:
        H_input = secret_imgv
    # ********************************************
    #cover_imgv = Hnet_C(cover_img)

    out_1, out_2, out_3, out_4, out_5, itm_secret_img = Hnet_C(input_1, input_2, input_3)
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
        print('opt.num_training', opt.num_training)
        #itm_secret_img = itm_secret_img.repeat(opt.num_training, 1, 1, 1)
        #print(itm_secret_img.shape)
        #***************
        #container_img = itm_secret_img + cover_imgv
        #container_img = Hnet_C(cover_img, out_dct_1, out_dct_2, out_dct_3, out_dct_4, itm_secret_img)
        #**************


    #rev_secret_img = Rnet(container_img, S_S)
    rev_secret_img = Rnet_S(container_img)
    rev_cover_img = Rnet_C(container_img)
    deta_secret = secret_imgv_nh - rev_secret_img
    deta_cover = cover_imgv - rev_cover_img
    F_input = torch.cat((deta_cover, deta_secret), dim=1)
    container_3, container_2, final_container = Hnet_F(F_input, out_1, out_2, out_3, out_4, out_5, itm_secret_img)
    revsecret3, revsecret2,final_rev_secret = RUnet_3(final_container)
    errH = criterion(final_container, cover_imgv)  # Hiding net
    errH_3 = criterion(container_3, cover_3)  # Hiding net
    errH_2 = criterion(container_2, cover_2)  # Hiding net
    errR_3 = criterion(revsecret3, secret_3)  # Reveal net
    errR_2 = criterion(revsecret2, secret_2)  # Reveal net
    errR = criterion(final_rev_secret, secret_imgv_nh)  # Reveal net

    # L1 metric
    diffH = (final_container - cover_imgv).abs().mean() * 255
    diffR = (final_rev_secret - secret_imgv_nh).abs().mean() * 255

    diffH_3 = (container_3 - cover_3).abs().mean() * 255
    diffR_3 = (revsecret3 - secret_3).abs().mean() * 255
    diffH_2 = (container_2 - cover_2).abs().mean() * 255
    diffR_2 = (revsecret2 - secret_2).abs().mean() * 255

    return cover_imgv, final_container, secret_imgv_nh, final_rev_secret, errH, errH_3, errH_2, errR, errR_3, errR_2, diffH, diffH_3, diffH_2, diffR, diffR_3, diffR_2

#train(train_loader, epoch, Hnet_C=Hnet_C, Hnet_F=Hnet_F, Rnet_C=Rnet_C, Rnet_S=Rnet_S, RUnet_3=RUnet_3, criterion=criterion)
def train(train_loader, epoch, Hnet_C, Hnet_F, Rnet_C, Rnet_S, RUnet_3, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses2 = AverageMeter()
    Rlosses2 = AverageMeter()
    Hlosses3 = AverageMeter()
    Rlosses3 = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    SumLosses = AverageMeter()
    Hdiff2 = AverageMeter()
    Rdiff2 = AverageMeter()
    Hdiff3 = AverageMeter()
    Rdiff3 = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    # Switch to train mode
    Hnet_C.train()
    Hnet_F.train()
    Rnet_C.train()
    Rnet_S.train()
    RUnet_3.train()

    start_time = time.time()

    for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(train_loader, 0):

        data_time.update(time.time() - start_time)
#cover_imgv, final_container, secret_imgv_nh, final_rev_secret, errH, errH_3, errH_2, errR, errR_3, errR_2, diffH, diffH_3, diffH_2, diffR, diffR_3, diffR_2
        cover_imgv, final_container, secret_imgv_nh, final_rev_secret, errH, errH_3, errH_2, errR, errR_3, errR_2, diffH, diffH_3, diffH_2, diffR, diffR_3, diffR_2 \
            = forward_pass(secret_img, secret_target, cover_img, cover_target, Hnet_C, Hnet_F, Rnet_C, Rnet_S, RUnet_3, criterion)

        Hlosses.update(errH.item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        Rlosses.update(errR.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        Hdiff.update(diffH.item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Rdiff.update(diffR.item(), opt.bs_secret * opt.num_secret * opt.num_training)

        Hlosses2.update(errH_2.item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        Rlosses2.update(errR_2.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        Hdiff2.update(diffH_2.item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Rdiff2.update(diffR_2.item(), opt.bs_secret * opt.num_secret * opt.num_training)

        Hlosses3.update(errH_3.item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        Rlosses3.update(errR_3.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        Hdiff3.update(diffH_3.item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Rdiff3.update(diffR_3.item(), opt.bs_secret * opt.num_secret * opt.num_training)


        # Loss, backprop, and optimization step
        betaerrR_secret = opt.beta * errR
        err_sum = opt.alpha*errH + opt.alpha*0.1*errH_3 + opt.alpha*0.1*errH_2 + betaerrR_secret + 0.1*opt.beta*errR_3 + 0.1*opt.beta*errR_2
        optimizer.zero_grad()
        err_sum.backward()
        optimizer.step()

        # Time spent on one batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\tLoss_H: %.6f Loss_R: %.6f L1_H: %.4f L1_R: %.4f Loss_H2: %.6f Loss_R2: %.6f L1_H2: %.4f L1_R2: %.4f Loss_H3: %.6f Loss_R3: %.6f L1_H3: %.4f L1_R3: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.epochs, i, opt.iters_per_epoch,
            Hlosses.val, Rlosses.val, Hdiff.val, Rdiff.val, Hlosses2.val, Rlosses2.val, Hdiff2.val, Rdiff2.val, Hlosses3.val, Rlosses3.val, Hdiff3.val, Rdiff3.val, data_time.val, batch_time.val)

        if i % opt.logFrequency == 0:
            print(log)



        if i == opt.iters_per_epoch - 1:
            break

    # To save the last batch only


    epoch_log = "Training[%d] Hloss=%.6f\t Rloss=%.6f\t Hdiff=%.4f\t Rdiff=%.4f\t Hloss2=%.6f\t Rloss2=%.6f\t Hdiff2=%.4f\t Rdiff2=%.4f\tHloss2=%.6f\t Rloss2=%.6f\t Hdiff2=%.4f\t Rdiff2=%.4f\tlr= %.6f\t Epoch time= %.4f" % (
    epoch, Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg, Hlosses2.avg, Rlosses2.avg, Hdiff2.avg, Rdiff2.avg, Hlosses3.avg, Rlosses3.avg, Hdiff3.avg, Rdiff3.avg,optimizer.param_groups[0]['lr'], batch_time.sum)
    print_log(epoch_log, logPath)

    if not opt.debug:
        writer.add_scalar("lr/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
        writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
        writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
        writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)
        writer.add_scalar('train/H_diff', Hdiff.avg, epoch)
        writer.add_scalar('train/R_diff', Rdiff.avg, epoch)


def validation(val_loader, epoch, Hnet_C, Hnet_F, Rnet_C, Rnet_S, RUnet_3, criterion):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Hnet_C.eval()
    Hnet_F.eval()
    Rnet_C.eval()
    Rnet_S.eval()
    RUnet_3.eval()
    batch_time = AverageMeter()
    Hlosses2 = AverageMeter()
    Rlosses2 = AverageMeter()
    Hlosses3 = AverageMeter()
    Rlosses3 = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    SumLosses = AverageMeter()
    Hdiff2 = AverageMeter()
    Rdiff2 = AverageMeter()
    Hdiff3 = AverageMeter()
    Rdiff3 = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(val_loader, 0):

        cover_imgv, final_container, secret_imgv_nh, final_rev_secret, errH, errH_3, errH_2, errR, errR_3, errR_2, diffH, diffH_3, diffH_2, diffR, diffR_3, diffR_2 \
            = forward_pass(secret_img, secret_target, cover_img, cover_target, Hnet_C, Hnet_F, Rnet_C, Rnet_S, RUnet_3, criterion, val_cover=1)

        Hlosses.update(errH.item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        Rlosses.update(errR.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        Hdiff.update(diffH.item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Rdiff.update(diffR.item(), opt.bs_secret * opt.num_secret * opt.num_training)

        Hlosses2.update(errH_2.item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        Rlosses2.update(errR_2.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        Hdiff2.update(diffH_2.item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Rdiff2.update(diffR_2.item(), opt.bs_secret * opt.num_secret * opt.num_training)

        Hlosses3.update(errH_3.item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        Rlosses3.update(errR_3.item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        Hdiff3.update(diffH_3.item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Rdiff3.update(diffR_3.item(), opt.bs_secret * opt.num_secret * opt.num_training)

        if opt.num_secret >= 6:
            i_total = 80
        else:
            i_total = 200
        if i == i_total - 1:
            break

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Hdiff = %.6f\t val_Rdiff=%.2f\t val_Hloss2 = %.6f\t val_Rloss2 = %.6f\t val_Hdiff2 = %.6f\t val_Rdiff2=%.2f\t val_Hloss3 = %.6f\t val_Rloss3 = %.6f\t val_Hdiff3 = %.6f\t val_Rdiff3=%.2f\t batch time=%.2f" % (
            epoch, Hlosses.val, Rlosses.val, Hdiff.val, Rdiff.val, Hlosses2.val, Rlosses2.val, Hdiff2.val, Rdiff2.val, Hlosses3.val, Rlosses3.val, Hdiff3.val, Rdiff3.val, batch_time.val)
        if i % opt.logFrequency == 0:
            print(val_log)

    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Hdiff = %.4f\t val_Rdiff=%.4f\t validation time=%.2f" % (
        epoch, Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg, batch_time.sum)
    print_log(val_log, logPath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
        writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
        writer.add_scalar('validation/H_diff_avg', Hdiff.avg, epoch)
        writer.add_scalar('validation/R_diff_avg', Rdiff.avg, epoch)

    print(
        "#################################################### validation end ########################################################")
    return Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg

def analysis(val_loader, epoch, Hnet_C, Hnet_F, Rnet_C, Rnet_S, RUnet_3, criterion):
    print(
        "#################################################### analysis begin ########################################################")

    Hnet_C.eval()
    Hnet_F.eval()
    Rnet_C.eval()
    Rnet_S.eval()
    RUnet_3.eval()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    psnr_C = AverageMeter()
    psnr_S = AverageMeter()
    ssim_C = AverageMeter()
    ssim_S = AverageMeter()
    lpips_C = AverageMeter()
    lpips_S = AverageMeter()

    #HnetD.eval()
    #RnetD.eval()
    import warnings
    warnings.filterwarnings("ignore")
    sum_psnr_c = 0
    sum_ssim_c = 0
    sum_lpips_c =0
    sum_psnr_s = 0
    sum_ssim_s = 0
    sum_lpips_s = 0

    for ii, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(val_loader, 0):

        ####################################### Cover Agnostic #######################################
        print('*********************')
        '''cover_imgv, final_container, secret_imgv_nh, final_rev_secret, errH, errH_3, errH_2, errR, errR_3, errR_2, diffH, diffH_3, diffH_2, diffR, diffR_3, diffR_2 \
            = forward_pass(secret_img, secret_target, cover_img, cover_target, Hnet_C, Hnet_F, Rnet_C, Rnet_S, RUnet_3, criterion, val_cover=1)'''
        cover_imgv, final_container, secret_imgv_nh, final_rev_secret, errH, errH_3, errH_2, errR, errR_3, errR_2, diffH, diffH_3, diffH_2, diffR, diffR_3, diffR_2 \
            = forward_pass(secret_img, secret_target, cover_img, cover_target, Hnet_C, Hnet_F, Rnet_C, Rnet_S, RUnet_3,
                           criterion, val_cover=1)
        secret_encoded = final_container - cover_imgv



        N, _, _, _ = final_rev_secret.shape

        cover_img_numpy = cover_imgv.clone().cpu().detach().numpy()
        container_img_numpy = final_container.clone().cpu().detach().numpy()

        cover_img_numpy = cover_img_numpy.transpose(0, 2, 3, 1)
        container_img_numpy = container_img_numpy.transpose(0, 2, 3, 1)

        rev_secret_numpy = final_rev_secret.cpu().detach().numpy()
        secret_img_numpy = secret_imgv_nh.cpu().detach().numpy()

        rev_secret_numpy = rev_secret_numpy.transpose(0, 2, 3, 1)
        secret_img_numpy = secret_img_numpy.transpose(0, 2, 3, 1)

        # PSNR
        print("Cover Agnostic")

        print("Secret APD C:", diffH.item())

        psnr_c = np.zeros((N, 3))
        for i in range(N):
            psnr_c[i, 0] = PSNR(cover_img_numpy[i, :, :, 0], container_img_numpy[i, :, :, 0])
            psnr_c[i, 1] = PSNR(cover_img_numpy[i, :, :, 1], container_img_numpy[i, :, :, 1])
            psnr_c[i, 2] = PSNR(cover_img_numpy[i, :, :, 2], container_img_numpy[i, :, :, 2])
        print("Avg. PSNR C:", psnr_c.mean().item())

        # SSIM
        ssim_c = np.zeros(N)
        for i in range(N):
            ssim_c[i] = SSIM(cover_img_numpy[i], container_img_numpy[i], multichannel=True)
        print("Avg. SSIM C:", ssim_c.mean().item())

        # LPIPS
        import PerceptualSimilarity.models
        model = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        lpips_c = model.forward(cover_imgv, final_container)
        print("Avg. LPIPS C:", lpips_c.mean().item())

        print("Secret APD S:", diffR.item())

        psnr_s = np.zeros(N)
        for i in range(N):
            psnr_s[i] = PSNR(secret_img_numpy[i], rev_secret_numpy[i])
        print("Avg. PSNR S:", psnr_s.mean().item())

        # SSIM
        ssim_s = np.zeros(N)
        for i in range(N):
            ssim_s[i] = SSIM(secret_img_numpy[i], rev_secret_numpy[i], multichannel=True)
        print("Avg. SSIM S:", ssim_s.mean().item())

        # LPIPS
        import PerceptualSimilarity.models
        model = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        secret_imgv_nh_2 = secret_imgv_nh.view(-1,3,128,128)
        final_rev_secret_2 = final_rev_secret.view(-1, 3, 128, 128)

        lpips_s = model.forward(secret_imgv_nh_2, final_rev_secret_2)
        print("Avg. LPIPS S:", lpips_s.mean().item())

        print("*******DONE!**********")

        #break
        lpips_S.update(lpips_s.mean().item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        psnr_S.update(psnr_s.mean().item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        ssim_S.update(ssim_s.mean().item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Rdiff.update(diffR.item(), opt.bs_secret * opt.num_secret * opt.num_training)
        lpips_C.update(lpips_c.mean().item(), opt.bs_secret * opt.num_cover * opt.num_training)  # H loss
        psnr_C.update(psnr_c.mean().item(), opt.bs_secret * opt.num_secret * opt.num_training)  # R loss
        ssim_C.update(ssim_c.mean().item(), opt.bs_secret * opt.num_cover * opt.num_training)
        Hdiff.update(diffH.item(), opt.bs_secret * opt.num_secret * opt.num_training)

        if opt.num_secret >= 6:
            i_total = 80
        else:
            i_total = 200
        if ii == i_total - 1:
            break
    print('Hdiff.avg, Rdiff.avg', Hdiff.avg, Rdiff.avg)

    print('Hdiff.avg', Hdiff.avg, 'psnr_c.avg', psnr_C.avg, 'ssim_c.avg', ssim_C.avg, 'lpips_c.avg',lpips_C.avg)
    print('Rdiff.avg', Rdiff.avg, 'psnr_s.avg', psnr_S.avg, 'ssim_s.avg', ssim_S.avg, 'lpips_s.avg',lpips_S.avg)
    print("*******DONE!**********")
#def analysis(val_loader, epoch, Hnet, Rnet, HnetD, RnetD, criterion):




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