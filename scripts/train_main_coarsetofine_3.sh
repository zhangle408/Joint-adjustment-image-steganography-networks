#!/usr/bin/env bash


#---in PyramidNet--------------------------------------------
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_coarsetofine_3.py \
--imageSize=128 \
--bs_secret=44 \
--num_training=1 \
--num_secret=2 \
--num_cover=1 \
--channel_cover=3 \
--channel_secret=3 \
--norm='batch' \
--epochs=65 \
--loss='l2' \
--alpha=1 \
--beta=0.75 \
--cover_dependent=1 \
--remark='main_coarsetofine_3' 
