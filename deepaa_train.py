#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train deep neural model.
"""
import time
from data_utils import BatchGenerator
from deepaa import DeepAA
import pickle
import numpy as np


def read_data_info(datapath):
    """
    return number of class and data
    :param datapath:
    :return:
    """
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    label = data['label']
    image = data['image']
    num_class = np.max(label) + 1
    num_data = image.shape[0]
    image_shape = image.shape[1:3]
    return num_class, num_data, image_shape


startTime = time.time()

# parameters パラメータ
datapath = 'Data/traindata.dump' # TODO change format
savepath = 'model_test'
DROPOUT = 0.5
BATCHSIZE = 128
EPOCHSIZE = 100
WEIGHTDECAY = 0.001

# Build a model モデル組み立て
num_class, num_data, image_shape = read_data_info(datapath)
print('number of class: ', num_class)
print('number of data: ', num_data)
print('shape of image: ', image_shape)
model = DeepAA(num_class, DROPOUT, WEIGHTDECAY, image_shape)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training 学習
num_batch_per_epoch = num_data // BATCHSIZE
gen = BatchGenerator(datapath, BATCHSIZE, num_batch_per_epoch)
print('training...')
model.fit_generator(gen, BATCHSIZE*num_batch_per_epoch, EPOCHSIZE, verbose=1)  # starts training

# save the model 保存
# /model 内にフォルダを作成してそこに保存
open(savepath+'.json', 'w').write(model.to_json())
model.save_weights(savepath+'_weight.hdf')

print('training end. %.1fsec' % (time.time() - startTime))