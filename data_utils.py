import numpy as np
from PIL import Image, ImageOps


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


    def add_mergin(self, h=18,w = 16):
        image = self.array
        new_image = np.ones((image.shape[0]+ 2 * h, image.shape[1] + 2 * w))
        new_image = new_image*255
        new_image[h:-h, w:-w] = image
        return new_image


    def gray_scale(self):
        image = Image.fromarray(self.array)
        image = ImageOps.grayscale(image)
        return np.asarray(image)