DeepAA
====

This is convolutional neural networks generating ASCII art.
This repositry is under the construction.

![image sample](https://github.com/OsciiArt/DeepAA/blob/master/sample%20images/images%20generated%20with%20CNN/21%20generated.png)

## Requirements

+ TensorFlow (1.3.0)
+ Keras (2.0.8)
+ NumPy (1.13.3)
+ Pillow (4.2.1)
+ Pandas (0.18.0)
+ Scikit-learn (0.19.0)
+ model's weight (download it from [here](https://drive.google.com/open?id=0B90WglS_AQWebjBleG5uRXpmbUE) and place it in dir `model`.)
## How to use
please change the line 15 of `output.py `

```
image_path = 'sample images/original images/21 original.png' # put the path of the image that you convert.
```
into the path of image file that you use.
You should use a grayscale line image.

then run `output.py `.
converted images will be output at `output/ `.

## License
The pre-trained models and the other files we have provided are licensed under the MIT License.
