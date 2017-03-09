import numpy as np
from PIL import Image, ImageOps
import pickle
from keras.utils import np_utils


class AsciiArt(list):
    """
    class of ASCII Art. List of Strings
    """
    def __init__(self, path=''):
        """
        path(txtファイル)から読み込み
        :param path:
        :return:
        """
        if path == '':
            list.__init__(self)
            self[:] = []
        else:
            list.__init__(self)
            text = open(path)
            text_list = []
            for line in text:
                text_list.append(line.strip('\n'))
            self[:] = text_list

    def __repr__(self):
        """
        print文を定義する
        ちなみにprint()でなくstr(), repr()だと省略せずに中身が表示される
        :return:
        """
        str = ''
        for item in self:
            str += item + '\n'
        return str

    def total_char(self):
        """
        総文字数を返す
        :return: int 総文字数
        """
        n = 0
        for i in range(len(self)):
            n += len(self[i])
        return n

    def y(self, line_height=18):
        return len(self)*line_height

    def x(self, char_list):
        """
        AAの横幅を返す
        :param char_list:
        :return:
        """
        x = 0
        for key, item in enumerate(self):
            x = max(x, self.line_width(key, char_list))
        return x

    def line_width(self, i, char_list):
        """
        i行の横幅を計算
        :param i: int 行番号
        :return:
        """
        str = self[i]
        total_width = 0
        for char in str:
            total_width += char_list.width(char)
        return total_width

    def line_image(self, i, char_list, line_height=18):
        """
        一行文の文字列をndarrayに変換
        ndarrayはグレイスケールの画像の形式 (0～255)

        :param i:
        :param char_list: CHarList
        :param line_height:
        :return:
        """
        width = self.line_width(i, char_list)
        height = line_height
        str = self[i]
        x = 0
        array = np.zeros((height, width))
        for char in str:
            char_width = char_list.width(char)
            array[:16, x:x+char_width] = char_list.array(char)
            x += char_width
        array = 255-255*array
        return array

    def image(self, char_list, line_height=18):
        """

        :param char_list:
        :param line_height:
        :return:
        """
        width = self.x(char_list)
        height = self.y()
        image = np.ones((height, width))*255
        for key, item in enumerate(self):
            y_start  =key * line_height
            y_end = (key + 1) * line_height
            x_start = 0
            x_end = self.line_width(key, char_list)
            image[y_start:y_end, x_start:x_end] = self.line_image(key, char_list)
        return image

    def line_width_list(self, line, char_list, line_height=18):
        """
        一行の文字列の各文字のxy座標のリストを返す
        :param line: 変換する行番号
        :param char_list: 文字幅を参照する文字リスト
        :param line_height: 行の高さ
        :return: ndarray. shape = (文字数, 2) 要素 = (何文字目?, y座標, x座標)
        """
        str = self[line]
        point_x = 0
        point_y = line * line_height
        t = np.zeros((len(str),2), dtype=int)
        for key, item in enumerate(str):
            t[key] = (point_y, point_x)
            point_x += char_list.width(item)
        return t

    def width_list(self, char_list, line_height=18):
        """
        文字列の各文字のxy座標のリストを返す
        :param char_list: 文字幅を参照する文字リスト
        :param line_height: 行の高さ
        :return: list = [ndarray, ndarray]. shape = (行数, 文字数, 2) 要素 = (行番号, 何文字目?, y座標, x座標)
        """
        width_list = np.empty((0,2), dtype=int)
        for i in range(len(self)):
            line_width_list = self.line_width_list(i, char_list, line_height)
            width_list = np.r_[width_list, line_width_list]
        return width_list

    def line_index_list(self, line, char_list):
        """
        選択した行の文字列の文字番号リストを返す
        :param line:
        :param char_list:
        :return:
        """
        str = self[line]
        t = np.empty(len(str), dtype=np.int)
        for i in range(len(str)):
            #print('char', str[i])
            t[i] = char_list.index(str[i])
            #print(t[i])
        return t

    def index_list(self, char_list):
        """
        AAの各文字の文字番号リストを返す
        :param char_list:
        :return:
        """
        t = np.empty(0, dtype=np.int)
        for line in range(len(self)):
            t = np.r_[t, self.line_index_list(line, char_list)]
        return t

    def as_one_array(self):
        """
        AAを一繋ぎのlistに変換
        :return:
        """
        t = []
        for line in self:
            t.extend(line)
        return t


class BaseImage(object):
    """
    変換元画像
    """
    def __init__(self, path):
        """
        元画像を読み込む
        :param path:
        :param array: np.ndarray
        :param line_height: int
        :return:
        """
        image = Image.open(path)
        array = np.asarray(image)
        self.array = np.array(array)


    def scale_image(self, new_width):
        """
        元画像の横幅を変更
        アスペクト比を維持する
        :param image: 元画像のndarray
        :param width: 新しい横幅
        :return: 横幅を修正したndarray
        """
        image = self.array
        original_width = image.shape[1]
        original_height = image.shape[0]
        aspect_ratio = original_height/float(original_width)
        new_height = int(aspect_ratio * new_width)
        image = Image.fromarray(image)
        new_image = image.resize((new_width, new_height), resample = Image.LANCZOS)
        return np.asarray(new_image)


    def add_mergin(self, mergin=[0, 0, 0, 0]):
        """
        padding = [up, down, left, right]
        :param padding:
        :return:
        """
        u, d, l, r = mergin
        image = self.array
        new_image = np.ones((image.shape[0]+ u + d, image.shape[1]+ l + r))
        new_image = new_image*255
        new_image[u:-d, l:-r] = image
        return new_image


    def gray_scale(self):
        image = Image.fromarray(self.array)
        image = ImageOps.grayscale(image)
        return np.asarray(image)


class CharList(list):
    """
    継承 list
    文字リスト
    """
    def array(self, key):
        """
        指定された文字のnparrayを返す
        :param key: string. 例) '鬱'
        :return: nparray
        """
        for item in self:
            if item[0] == key:
                return item[1]

    def width(self, char):
        """
        指定した文字の幅を返す
        :param char:
        :return:
        """
        for item in self:
            if item[0] == char:
                return item[1].shape[1]
        return 0


    def show(self, key):
        """
        指定された文字のnparrayをprintする
        :param key: string. 例) '鬱'
        """
        flag = 0
        for item in self:
            if item[0] == key:
                new_item = item[1].astype(int)
                print(new_item)
                flag = 1
        if not flag:
            print('No such a key!')

    def index(self, char):
        """
        指定された文字の番号(リストのkey)を返す
        もし見つからなかったら-1を返す
        eg) char_list.index(' ') = 0 要素番号0
        :param char:
        :return: int
        """
        flag = 0
        for key, item in enumerate(self):
            #print(char, item[0])
            if item[0] == char:
                return key
        return -1

    def delete_NG_word(self, NGword=['■','●','▲','▼','★','◆']):
        """
        文字リストからNGワードを除去
        :param char_list: list. 文字リスト [('鬱', np(bool)]
        :param NGword: list. NGワードのリスト
        :return: list. NGワードを除去した文字リスト
        """
        new_list = CharList()
        for item in self:
            if item[0] not in NGword:
                new_list.append(item)
        return new_list

    def convert_to_FontList(self):
        """
        文字リストをフォントリストに変換
        :return: FontList
        """
        total = len(self) #総文字数

        #fontのみのアレイ, 文字のみのアレイ, 幅のみのアレイを作成
        total_w_array = np.empty(total) #文字幅を抽出
        total_font_list = np.zeros((total, 16, 16), dtype = bool) #arrayのみ抽出
        total_char_list = np.empty((total, 1), dtype= 'U16') #文字を抽出
        for key, item in enumerate(self):
            total_w_array[key] = item[1].shape[1]
            total_font_list[key, :, :total_w_array[key]] = item[1]
            total_char_list[key] = item[0]

        #ブールインデックスで幅ごとのarrayに仕分け
        new_font_list = FontList([0]*16)
        #print(len(new_font_list))
        new_char_list = [0]*16
        #print(len(new_char_list))
        for i in range(16):
            new_font_list[i] = total_font_list[total_w_array == i + 1, :, :i+1] #幅がi+1の文字のみピックアップ
            new_char_list[i] = total_char_list[total_w_array == i + 1]
            new_char_list[i] = new_char_list[i].reshape(new_char_list[i].shape[0])
        new_font_list.char = new_char_list
        #print('fontlist.char.shape', new_font_list.char[0].shape)
        return new_font_list


def BatchGenerator(datapath, batch_size, num_batch_per_epoch):
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    x1 = data['image']
    y = data['label']
    onehot = np_utils.to_categorical(y)
    idx = 0
    while True:
        if idx == 0:
            perm = np.arange(x1.shape[0])
            np.random.shuffle(perm)
        x1batch = x1[perm[idx:idx+batch_size]].astype(np.float32)/255
        ybatch = onehot[perm[idx:idx+batch_size]].astype(np.float32)
        if idx + batch_size > batch_size * num_batch_per_epoch:
            idx = 0
        else:
            idx += batch_size
        yield (x1batch, ybatch)