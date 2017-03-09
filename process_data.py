"""
Process AST file of ASCII Art into train data.
AST file is a text file contains ASCII Art with separater '[AA][]\n'
"""

import numpy as np
import pickle
from PIL import Image
import glob, os
from data_utils import AsciiArt, CharList


def Ast2Txt(readpath, savepath, header=''):
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    print('convert AST to txt. save path: ', savepath)
    # read AST file
    txt = open(readpath)
    txtlist = []
    for line in txt:
        txtlist.append(line)
    borderlist = []
    for key, item in enumerate(txtlist):
        if item == '[AA][]\n':
            borderlist.append(key)

    # convert
    for i in range(len(borderlist)):
        filename = header + '{0:04d}'.format(i+1) + ".txt"
        if not i == len(borderlist) - 1:  # if not the line is the end of the AST
            with open(savepath  + filename, 'w') as f:
                f.writelines(txtlist[borderlist[i] + 1:borderlist[i+1]])
        else:
            with open(savepath + filename, 'w') as f:
                f.writelines(txtlist[borderlist[i] + 1:])
    print(len(borderlist), 'text files saved.', )


def AA2Image(readpath, savepath, header, font_data):
    """
    output files in the folder as image
    :param readpath: folder path to read
    :param savepath: path to save image
    :param dash: str
    :return:
    """
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    print('convert txt to png. save path: ', savepath)

    files = glob.glob(readpath+'*.txt')

    for file in files:
        ascii_art = AsciiArt(file)
        ascii_art_image = ascii_art.image(font_data)
        filename = header + os.path.basename(file)[:-4] + '.png'
        ascii_art_image = Image.fromarray(ascii_art_image)
        ascii_art_image = ascii_art_image.convert('L')
        ascii_art_image.save(savepath  + filename)
        print('saved ', filename)


def gen_char_list(readpath, savepath, font_data):
    """
    output the list contains charcter information
    column 1: character
    column 2; character's width
    column 3: character's frequency
    :param readpath:
    :param savepath:
    :param font_data:
    :return:
    """
    # txt内の文字をカウントしていく
    filelist = os.listdir(readpath)
    chardict = {}
    for filepath in filelist:
        file = open(readpath + filepath)
        for line in file:
            for char in line.replace('\n', ''):
                if char in chardict:
                    chardict[char] += 1
                else:
                    chardict[char] = 1

    # カウント数でソートして多い順にラベル化
    # TODO if frequency is same, sorted by number of shift-jis code
    char_list = []
    for key, value in sorted(chardict.items(), key=lambda x: x[1]):
        char_list.append([key, font_data.width(key), value])
    char_list.reverse()

    for key, item in enumerate(char_list): # TODO remove if bag fixed
        if item[1]==0:
            char_list[key][1] = 16

    for item in char_list:
        print(item)
    # 保存
    with open(savepath, 'wb') as f: # TODO CSV形式化
        pickle.dump(char_list, f)

    print('saved ', savepath, ' number of character is ', len(char_list))


def image_slice(base_image, x, y, slice_mergin):
    Im_Campus_slice = Image.new('RGBA', (16+slice_mergin*2, 16+slice_mergin*2), (255, 255, 255, 0))
    Im_B = base_image.convert('RGBA')
    #Im_B.show()
    Im_B_slice = Im_B.crop((x - slice_mergin, y - slice_mergin, x + 16 + slice_mergin, y + 16 + slice_mergin))
    Im_B_slice = Image.alpha_composite(Im_Campus_slice, Im_B_slice)
    #Im_B_slice.show()
    Im_B_slice = Im_B_slice.convert('L')
    Im_B_slice = np.asarray(Im_B_slice, dtype=np.uint8)
    #print(Im_B_slice)
    return Im_B_slice


def PNG2Np(imgpath, AApath, font_data, char_list,  x=0, y=0, z='', slice_mergin = 8):
    """

    :param imgpath: 画像パス
    :param AApath: AAパス
    :param x: ｘ座標
    :param y: ｙ座標
    :param z: 倍率
    :param charlist: 文字のライブラリ
    :return:
    """
    # TODO rewrite more readable
    # TODO fasten
    print("convert ", imgpath, 'with ', AApath)

    # 読み込み
    ascii_art = AsciiArt(AApath)
    Im_B = Image.open(imgpath)
    Im_B = Im_B.convert('L')
    if z == '': z = ascii_art.x(font_data) / Im_B.size[0]
    Im_B = Im_B.resize((int(Im_B.width*z),int(Im_B.height*z)), resample=Image.LANCZOS)

    # AAの文字のindexのlistを作成
    aa_list = ascii_art.as_one_array()
    label_list =np.ones([len(aa_list)], dtype=np.uint16)

    for i in range(label_list.shape[0]):
        for j in range(len(char_list)):
            # print(aa_list[i])
            # print(char_list[j])
            if aa_list[i] == char_list[j][0]:
                label_list[i] = j
                break # print(char_list[j][0])

    label_list = label_list.reshape([label_list.shape[0], 1])
    # 全文字の座標リスト作成, x,yで補正する
    char_position_list = ascii_art.width_list(font_data)
    linebegin_list = np.zeros(len(aa_list), dtype=np.bool)
    for i in range(len(aa_list)):
        if char_position_list[i,1] == 0:
            linebegin_list[i] = 1

    char_position_list = char_position_list - np.array([y, x])


    #各座標について対応する元画像のスライスを取り出しリスト化
    # image_list: ndarray, AAの総文字数 x 32 x 32
    # 縦 = マージンx2 + 16
    image_list = np.empty((ascii_art.total_char(), slice_mergin*2+16, slice_mergin*2+16), dtype=np.uint8)
    for i in range(image_list.shape[0]):
        slice = image_slice(Im_B, char_position_list[i,1], char_position_list[i,0], slice_mergin)
        image_list[i] = slice
    return image_list, label_list, linebegin_list


def gen_train_data(readpath, savepath, font_data, char_list):
    # TODO rewrite more readable
    print('generating training data...')
    imgFolderPath = readpath + '/image/'
    AAFolderPath = readpath + '/text/'
    labellist = []
    for item in font_data:
        labellist.append(font_data.index(item))

    image_lists = np.zeros([0, 64, 64], dtype=np.uint8)
    label_lists = np.zeros([0, 1], dtype=np.uint16)
    linebegin_lists = np.zeros([0,], dtype=np.bool)
    source_lists = []
    files = os.listdir(imgFolderPath)

    # convert each file
    for file in files:
        if not file.find(".png")==-1:
            imgPath = imgFolderPath + file
            AAPath = AAFolderPath + os.path.basename(imgPath)[-8:-4] + ".txt"
            image_list, label_list, linebegin_list = PNG2Np(imgPath, AAPath, font_data, char_list, slice_mergin = 24)
            image_lists = np.r_[image_lists, image_list]
            label_lists = np.r_[label_lists, label_list]
            linebegin_lists = np.r_[linebegin_lists, linebegin_list]
            source = imgPath
            source_lists += [source]*image_list.shape[0]

    dict = {
        'image': image_lists,
        'label': label_lists,
        'linebegin': linebegin_lists,
        'source': source_lists
    }

    # save
    with open(savepath, 'wb') as f:
        pickle.dump(dict, f)

    print('saved ', savepath, 'nuber of data: ', image_lists.shape[0])

def main():
    # parameters
    readpath = 'data/aa_sample.ast'
    savefolder = 'data/traindata'
    header = ''

    # processing
    # TODO pickle以外の形式に差し替え
    font_data_path = 'data/font_data.dump' # TODO sjisを網羅したものに差し替える
    with open(font_data_path, 'rb') as f:
        font_data = pickle.load(f)
    Ast2Txt(readpath, savefolder+'/text/', header)
    AA2Image(savefolder+'/text/', savefolder+'/image/', header, font_data)
    gen_char_list(savefolder+'/text/', savefolder+'/char_list.dump', font_data)
    # in this step images should be distorted somehow for generalizing data
    with open(savefolder+'/char_list.dump', 'rb') as f:
        char_list = pickle.load(f)
    gen_train_data(savefolder, savefolder+'/train_data.dump', font_data, char_list)


if __name__=='__main__': main()