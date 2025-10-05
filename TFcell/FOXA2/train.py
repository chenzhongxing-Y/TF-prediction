import pyBigWig
import argparse
import os
import sys
import numpy as np
import re
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
import tensorflow as tf
import scipy.io
import os

# 禁用所有GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print('tf-' + tf.__version__)

import random
from datetime import datetime
import unet
from unet import get_unet, FlowAttention  
# 配置TensorFlow维度顺序
tf.keras.backend.set_image_data_format('channels_last')

# 染色体及长度定义
chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=np.array([249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560])

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

# 模型及训练参数
size=2**11*5 # 10240
num_channel=6
num_sample=100000
batch_size=16

# 余弦衰减函数
def cosine_decay(epoch):
    lr_max = 1e-3  
    lr_min = 1e-5  
    epochs_total = 10  # 匹配训练总轮次
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / epochs_total * np.pi))

# 数据路径定义
path_computer='../../data/'
path1=path_computer + 'dna_bigwig/' # dna数据路径
path2=path_computer + 'dnase_bigwig/' # dnase数据路径
path3=path_computer + 'chip-seq gem peak/' # 标签数据路径

# 命令行参数解析
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-tf', '--transcription_factor', default='CTCF', type=str,
        help='transcript factor')
    parser.add_argument('-tr', '--train', default='K562', type=str,
        help='train cell type')
    parser.add_argument('-vali', '--validate', default='A549', type=str,
        help='validate cell type')
    parser.add_argument('-par', '--partition', default='1', type=str,
        help='chromasome parition')
    args = parser.parse_args()
    return args

args=get_args()

# 解析参数并打印
print(sys.argv)
the_tf=args.transcription_factor
cell_train=args.train
cell_vali=args.validate
par=args.partition 

# 训练/验证染色体划分
chr_train_all=['chr2','chr3','chr4','chr5','chr6','chr7','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr22','chrX']
ratio=0.5
np.random.seed(int(par))
np.random.shuffle(chr_train_all)
tmp=int(len(chr_train_all)*ratio)
chr_set1=chr_train_all[:tmp] # 训练集染色体
chr_set2=chr_train_all[tmp:] # 验证集染色体
print("训练集染色体:", chr_set1)
print("验证集染色体:", chr_set2)

# 模型名称定义及模型加载
name_model='weights_' + cell_train + '_' + cell_vali + '_' + par + '.h5'
model = unet.get_unet(the_lr=1e-4,num_class=1,num_channel=num_channel,size=size)
model.summary()

## 训练/验证集染色体采样索引生成
# 训练集索引
tmp=[]
for the_chr in chr_set1:
    tmp.append(chr_len[the_chr])
freq=np.rint(np.array(tmp)/sum(tmp)*1000).astype('int')
index_set1=np.array([])
for i in np.arange(len(chr_set1)):
    index_set1 = np.hstack((index_set1, np.array([chr_set1[i]] * freq[i])))
np.random.shuffle(index_set1)

# 验证集索引
tmp=[]
for the_chr in chr_set2:
    tmp.append(chr_len[the_chr])
freq=np.rint(np.array(tmp)/sum(tmp)*1000).astype('int')
index_set2=np.array([])
for i in np.arange(len(chr_set2)):
    index_set2 = np.hstack((index_set2, np.array([chr_set2[i]] * freq[i])))
np.random.shuffle(index_set2)

# 打开BigWig文件
list_dna=['A','C','G','T']
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
feature_avg=pyBigWig.open(path2 + 'avg.bigwig')
feature_train=pyBigWig.open(path2 + cell_train + '.bigwig')
# 打印验证集DNase文件路径
print(f"DNase验证文件路径: {os.path.abspath(path2 + cell_vali + '.bigwig')}")
feature_vali=pyBigWig.open(path2 + cell_vali + '.bigwig')
label_train=pyBigWig.open(path3 + the_tf + '_' + cell_train + '.bigwig')
label_vali=pyBigWig.open(path3 + the_tf + '_' + cell_vali + '.bigwig')

##### 数据增强参数
if_time=False
max_scale=1.15
min_scale=1
if_mag=True
max_mag=1.15
min_mag=0.9
if_flip=False

####################################
# 数据生成器
def generate_data(batch_size, if_train):
    i = 0
    j = 0
    while True:
        b = 0
        image_batch = []
        label_batch = []

        while b < batch_size:
            # 选择训练/验证数据
            if (if_train == 1):
                if i == len(index_set1):
                    i = 0
                    np.random.shuffle(index_set1)
                the_chr = index_set1[i]
                feature_bw = feature_train
                label_bw = label_train
                i += 1
            else:
                if j == len(index_set2):
                    j = 0
                    np.random.shuffle(index_set2)
                the_chr = index_set2[j]
                feature_bw = feature_vali
                label_bw = label_vali
                j += 1

            # 随机选择片段起始位置
            start = int(np.random.randint(0, chr_len[the_chr] - size, 1))
            end = start + size

            # 读取标签并处理NaN
            label = np.array(label_bw.values(the_chr, start, end))
            label = np.nan_to_num(label, nan=-1.0)  # 标签NaN转为-1

            # 读取图像特征（DNA+DNase+差异信号）
            image = np.zeros((num_channel, size))
            # DNA碱基通道（A/C/G/T）
            for k in np.arange(len(list_dna)):
                the_id = list_dna[k]
                dna_values = dict_dna[the_id].values(the_chr, start, end)
                image[k, :] = np.nan_to_num(dna_values, nan=0.0)  # DNA值NaN转为0
            # DNase信号通道
            feature_values = np.nan_to_num(feature_bw.values(the_chr, start, end), nan=0.0)
            image[4, :] = feature_values
            # DNase差异信号（当前细胞 - 平均）
            avg = np.nan_to_num(feature_avg.values(the_chr, start, end), nan=0.0)
            image[5, :] = image[4, :] - avg

            # 训练集数据增强
            if if_train == 1:
                # 信号幅度缩放增强
                if if_mag:  
                    rrr = random.random()
                    rrr_mag = rrr * (max_mag - min_mag) + min_mag
                    image[4, :] = image[4, :] * rrr_mag  # 仅缩放DNase信号
                # 水平翻转增强（50%概率）
                if np.random.rand() > 0.5:
                    image = np.fliplr(image)
                    label = np.flip(label)

            
            image = image.T  

            # 加入批次列表
            image_batch.append(image)
            label_batch.append(label.reshape(-1, 1))  
            b += 1

        
        image_batch = np.array(image_batch)
        label_batch = np.array(label_batch)
        yield image_batch, label_batch

# 训练回调函数
callbacks = [
    
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join('./', name_model),
        save_weights_only=False,
        monitor='val_loss',
    ),
    # 保存基于验证集Dice系数（val_dice_coef）的最佳模型
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_dice_' + name_model,
        save_weights_only=False, 
        monitor='val_dice_coef',  # 监控验证集Dice系数
        mode='max',               
        save_best_only=True       # 仅保存最佳模型
    ),
    #  余弦衰减学习率调度器
    LearningRateScheduler(cosine_decay, verbose=1)
]

# 模型训练
model.fit(
    generate_data(batch_size, True),    # 训练数据生成器
    steps_per_epoch=int(num_sample // batch_size),  # 每轮训练步数（总样本//批次大小）
    epochs=10,                          # 总训练轮次
    validation_data=generate_data(batch_size, False),  # 验证数据生成器
    validation_steps=int(num_sample // batch_size),    # 每轮验证步数
    callbacks=callbacks,                # 训练回调
    verbose=1                           # 打印训练过程（每轮进度+损失/指标）
)

# 关闭所有BigWig文件
for the_id in list_dna:
    dict_dna[the_id].close()
feature_avg.close()
feature_train.close()
feature_vali.close()
label_train.close()
label_vali.close()