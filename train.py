from keras.models import Model
from keras.layers import Dense, Activation, Reshape, Dropout, Embedding, Input, BatchNormalization
from keras.layers import Concatenate, Multiply, Conv2D, MaxPooling2D, Add, Flatten, GaussianNoise
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, \
    EarlyStopping, CSVLogger, ReduceLROnPlateau

import time
import numpy as np

np.random.seed(42)
import pandas as pd
from os import path
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from multiprocessing import Pool


def CBRD(inputs, filters=64, kernel_size=(3,3), droprate=0.5):
    x = Conv2D(filters, kernel_size, padding='same',
               kernel_initializer='random_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(droprate)(x)
    return x


def DBRD(inputs, units=4096, droprate=0.5):
    x = Dense(units)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(droprate)(x)
    return x

def CNN(input_shape=None, classes=1000):
    inputs = Input(shape=input_shape)

    # Block 1
    x = GaussianNoise(0.3)(inputs)
    x = CBRD(x, 64)
    x = CBRD(x, 64)
    x = MaxPooling2D()(x)

    # Block 2
    x = CBRD(x, 128)
    x = CBRD(x, 128)
    x = MaxPooling2D()(x)

    # Block 3
    x = CBRD(x, 256)
    x = CBRD(x, 256)
    x = CBRD(x, 256)
    x = MaxPooling2D()(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = DBRD(x, 4096)
    x = DBRD(x, 4096)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def add_mergin(img, mergin):
    if mergin!=0:
        img_new = np.ones([img.shape[0] + 2 * mergin, img.shape[1] + 2 * mergin], dtype=np.uint8) * 255
        img_new[mergin:-mergin, mergin:-mergin] = img
    else:
        img_new = img
    return img_new


def load_img(args):
    img_path, x, y, input_size, mergin, slide = args
    img = np.array(Image.open(img_path))
    if len(img.shape) == 3:
        img = img[:, :, 0]
    img = add_mergin(img, mergin)
    x += np.random.randint(-slide, slide+1)
    y += np.random.randint(-slide, slide+1)
    img = img[y:y + input_size, x:x + input_size]
    img = img.reshape([1, input_size, input_size, 1])
    # print(img_path, x, y, input_size, mergin )
    # print(input_size, img.shape)
    return img

def batch_generator(df, img_dir, input_size, batch_size, num_label, slide,
                    tail='line', shuffle=True):
    df = df.reset_index()
    batch_index = 0
    mergin = (input_size - 18) // 2 + 30
    n = df.shape[0]
    pool = Pool()
    while 1:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        index_array_batch = index_array[current_index: current_index + current_batch_size]
        batch_img_path = df['file_name'][index_array_batch].apply(
            lambda x: img_dir + x + tail + '.png').as_matrix()
        # print(batch_img_path)
        batch_coord_x = (df['x'][index_array_batch] + 30).as_matrix()
        batch_coord_y = (df['y'][index_array_batch] + 30).as_matrix()
        # print(batch_img_path[0], batch_coord_x[0], batch_coord_y[0], mergin)
        batch_x = pool.map(load_img,
                           [(batch_img_path[i],
                             batch_coord_x[i],
                             batch_coord_y[i],
                             input_size,
                             mergin,
                             slide)
                           for i in range(current_batch_size)])
        # print(batch_x[0].shape)
        batch_x = np.concatenate(batch_x, axis=0)
        batch_x = batch_x.astype(np.float32) / 255
        # print(batch_x.shape)

        batch_y = df['label'][index_array[current_index: current_index + current_batch_size]].as_matrix()
        batch_y = np.eye(num_label)[batch_y]

        yield batch_x, batch_y


def train_generator(df, img_dir, input_size, batch_size, num_label, slide,
                    tail='line', shuffle=True):
    gen_line = batch_generator(df, img_dir, input_size,
                               batch_size // 2, num_label, slide, tail="line_resize")
    gen_orig = batch_generator(df, img_dir, input_size,
                               batch_size // 2, num_label, slide, tail="orig")
    while True:
        batch1 = next(gen_line)
        batch2 = next(gen_orig)
        batch_x = np.concatenate([batch1[0], batch2[0]])
        batch_y = np.concatenate([batch1[1], batch2[1]])
        yield batch_x, batch_y


def train():
    # parameter
    num_epoch = 256
    batch_size = 64
    input_shape = [64,64,1]
    learning_rate = 0.001
    df_path = "data/data_500.csv"
    char_list_path = "data/char_list_500.csv"
    img_dir = "data/image_500/"

    # load text
    df = pd.read_csv(df_path, encoding="cp932")
    char_list = pd.read_csv(char_list_path, encoding="cp932")
    num_label = char_list[char_list['frequency']>=10].shape[0]
    # print(num_label)
    df = df[df['label']<num_label]
    df = df.reset_index()
    input_size = input_shape[0]
    slide = 1
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    gen = train_generator(df_train, img_dir,
                          input_size, batch_size, num_label, slide)
    gen_val = batch_generator(df_val, img_dir, input_size,
                              batch_size, num_label, 0,
                              tail="line_resize", shuffle=False)

    # build model
    model = CNN(input_shape=input_shape, classes=num_label)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # train
    nb_train = df_train.shape[0]
    nb_val = df_val.shape[0]
    nb_step = math.ceil(nb_train / batch_size)
    nb_val_step = math.ceil(nb_val / batch_size)

    format = "%H%M"
    ts = time.strftime(format)
    save_path = "model/" + path.splitext(__file__)[0] + "_" + ts

    json_string = model.to_json()
    with open(save_path + '_model.json', "w") as f:
        f.write(json_string)

    csv_logger = CSVLogger(save_path + '_log.csv', append=True)
    check_path = save_path + '_e{epoch:02d}_vl{val_loss:.5f}.hdf5'
    save_checkpoint = ModelCheckpoint(filepath=check_path, monitor='val_loss', save_best_only=True)
    lerning_rate_schedular = ReduceLROnPlateau(patience=8, min_lr=learning_rate * 0.00001)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=16,
                                   verbose=1,
                                   min_delta=1e-4,
                                   mode='min')
    Callbacks = [csv_logger,
                 save_checkpoint,
                 lerning_rate_schedular, early_stopping]
    model.fit_generator(gen,
                        steps_per_epoch=nb_step,
                        epochs=num_epoch,
                        validation_data=gen_val,
                        validation_steps=nb_val_step,
                        callbacks=Callbacks
                        )


if __name__ == "__main__":
    train()