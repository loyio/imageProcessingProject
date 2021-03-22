"""
 @Project: imageProcessing
 @Author: loyio
 @Date: 3/20/21
"""
from PIL import Image, ImageFilter
import numpy as np

if __name__ == '__main__':
    # greyscale
    imgfileOne = "Sample/1"
    sample_img = Image.open(imgfileOne+".jpg").convert('L')
    sample_img.save(imgfileOne+"_processed_gray.jpg")

    # Blur
    # sample_img = Image.open(imgfileOne+".jpg").filter(ImageFilter.BLUR)
    sample_img = Image.open(imgfileOne+".jpg").filter(ImageFilter.BoxBlur(5))
    sample_img.save(imgfileOne + "_processed_blur.jpg")

    # Paint
    imgfileTwo = "Sample/2"
    sample_img_ary = np.asarray(Image.open(imgfileTwo+".jpg").convert('L')).astype('float')

    depth = 10.
    grad_x, grad_y = np.gradient(sample_img_ary)
    grad_x = grad_x * depth / 100.
    grad_y = grad_y * depth / 100.

    A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
    uni_x = grad_x / A
    uni_y = grad_y / A
    uni_z = 1. / A

    vec_el = np.pi / 2.2
    vec_az = np.pi / 4.
    dx = np.cos(vec_el) * np.cos(vec_az)
    dy = np.cos(vec_el) * np.sin(vec_az)
    dz = np.sin(vec_el)

    sample_processed_ary = (255 * (dx*uni_x + dy*uni_y + dz*uni_z)).clip(0, 255)

    im = Image.fromarray(sample_processed_ary.astype('uint8'))
    im.save(imgfileTwo+"_processed_handpaint.jpg")