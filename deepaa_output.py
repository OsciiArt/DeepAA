#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像をアスキーアートに変換する
"""

from keras.models import model_from_json
import pickle
import numpy as np
import time
from data_utils import BaseImage

start_time = time.time()

# パラメータ
image_path = 'test_image.png' # 変換する画像を指定
image_width = 550 # 画像サイズの変換設定. 変換後の横幅を入力する. ''なら変換しない. 1st setting: 550
model_folder = 'model_01/' # 変換用のモデルを選択
multi_output = False # 1ピクセルずつ縦にずらして18個出力

# ニューラルネットモデル読み込み
json_string = open('model/' + model_folder + 'model.json').read()
model = model_from_json(json_string)
model.load_weights('model/' + model_folder+'weight.hdf')

# 出力用ファイル読み込み
# TODO csvファイル化
with open('model/' + model_folder + 'char_list.dump', 'rb') as f:
    char_list = pickle.load(f)

# 画像前処理
base_image = BaseImage(image_path)
base_image.array = base_image.gray_scale()
if not image_width=='':
    base_image.array = base_image.scale_image(image_width)
base_image.array = base_image.add_mergin(mergin=[24, 24+17, 24, 24+17])
base_image.array = base_image.array.astype(np.float32) / 255

# 変換
print('conversion start. image: %s, model: %s' % (image_path, model_folder))
lineNum = (base_image.array.shape[0] - 48) // 18
slide_range = 18 if multi_output==True else 1
for slide in range(slide_range):
    for i in range(lineNum):
        lineImage = base_image.array[i * 18: i * 18 + 64]
        lineCharList = ''
        start, end  = 0, 64
        penalty = 1 # 行頭半角スペース、連続半角スペースを禁止するためのフラグ
        while end <= base_image.array.shape[1]:
            patch = lineImage[:, start:end].reshape([1, 64, 64])
            y = model.predict(patch, 1)

            if penalty==1:
                y[:,1] = 0

            predict = np.argmax(y)
            char = char_list[predict][0]
            width = char_list[predict][1]
            lineCharList += char
            start += width
            end += width

            if predict==1:
                penalty = 1
            else:
                penalty = 0

        print(lineCharList)
    base_image.array = np.r_[np.ones([1, base_image.array.shape[1]]), base_image.array]




print('conversion end. %.1fsec' % (time.time() - start_time))