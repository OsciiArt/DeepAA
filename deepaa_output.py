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
image_width = 550 # 画像サイズの変換設定. 変換後の横幅を入力する. ''なら変換しない
model_path = 'model_01' # 変換用のモデルを選択

# ニューラルネットモデル読み込み
json_string = open('model/' + model_path + '.json').read()
model = model_from_json(json_string)
model.load_weights('model/' + model_path+'_weight.hdf')

# 出力用ファイル読み込み
label2widthlistpath = 'data/labal2width_01.dump'
with open(label2widthlistpath, 'rb') as f:
    label2widthlist = pickle.load(f)
label2charlistpath = 'data/labal2char_01.dump'
with open(label2charlistpath, 'rb') as f:
    label2charlist = pickle.load(f)

# 画像前処理
base_image = BaseImage(image_path)
base_image.array = base_image.gray_scale()
if not image_width=='':
    base_image.array = base_image.scale_image(image_width)
base_image.array = base_image.add_mergin(24, 24)
base_image.array = base_image.array.astype(np.float32) / 255

# 変換
print('conversion start. image: %s, model: %s' % (image_path, model_path))
lineNum = (base_image.array.shape[0] - 48) // 18
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
        char = label2charlist[predict]
        width = label2widthlist[predict]
        lineCharList += char
        start += width
        end += width

        if predict==1:
            penalty = 1
        else:
            penalty = 0

    print(lineCharList)

print('conversion end. %.1fsec' % (time.time() - start_time))