



from PIL import Image

from PIL import ImageEnhance

import os

import cv2

import numpy as np




def flip(root_path,img_name):   #翻转图像

    img = Image.open(os.path.join(root_path, img_name))

    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))

    return filp_img


def rotation(root_path, img_name,a):

    img = Image.open(os.path.join(root_path, img_name))

    rotation_img = img.rotate(a) #旋转角度

    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))

    return rotation_img


def randomColor(root_path, img_name): #随机颜色

    """

    对图像进行颜色抖动

    :param image: PIL的图像image

    :return: 有颜色色差的图像image

    """

    image = Image.open(os.path.join(root_path, img_name))

    random_factor = np.random.randint(0, 31) / 10.  # 随机因子

    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度

    random_factor = np.random.randint(10, 21) / 10.  # 随机因子

    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度

    random_factor = np.random.randint(10, 21) / 10.  # 随机因子

    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度

    random_factor = np.random.randint(0, 31) / 10.  # 随机因子

    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def contrastEnhancement(root_path, img_name):  # 对比度增强

    image = Image.open(os.path.join(root_path, img_name))

    enh_con = ImageEnhance.Contrast(image)

    contrast = 1.5

    image_contrasted = enh_con.enhance(contrast)

    return image_contrasted


def brightnessEnhancement(root_path,img_name):#亮度增强

    image = Image.open(os.path.join(root_path, img_name))

    enh_bri = ImageEnhance.Brightness(image)

    brightness = 1.5

    image_brightened = enh_bri.enhance(brightness)

    return image_brightened


def colorEnhancement(root_path,img_name):#颜色增强

    image = Image.open(os.path.join(root_path, img_name))

    enh_col = ImageEnhance.Color(image)

    color = 1.5

    image_colored = enh_col.enhance(color)

    return image_colored



imageDir="./images" #要改变的图片的路径文件夹

saveDir="./results"   #要保存的图片的路径文件夹

i=0

for name in os.listdir(imageDir):

    i=i+1
    # 可用flip（翻转），rotation（旋转），randomColor（随机颜色），contrastEnhancement（对比增强），brightnessEnhancement（亮度增强），colorEnhancement（颜色增强）
    saveName="flip"+str(i)+".jpg"
    saveImage=flip(imageDir,name)
    saveImage.save(os.path.join(saveDir, saveName))

    saveName="rotationa"+str(i)+".jpg"
    saveImage=rotation(imageDir,name,120)
    saveImage.save(os.path.join(saveDir, saveName))

    saveName="rotationb"+str(i)+".jpg"
    saveImage=rotation(imageDir,name,240)
    saveImage.save(os.path.join(saveDir, saveName))

    saveName="randomColor"+str(i)+".jpg"
    saveImage=randomColor(imageDir,name)
    saveImage.save(os.path.join(saveDir, saveName))

    saveName="contrastEnhancement"+str(i)+".jpg"
    saveImage=contrastEnhancement(imageDir,name)
    saveImage.save(os.path.join(saveDir, saveName))

    saveName="brightnessEnhancement"+str(i)+".jpg"
    saveImage=brightnessEnhancement(imageDir,name)
    saveImage.save(os.path.join(saveDir, saveName))

    saveName="colorEnhancement"+str(i)+".jpg"
    saveImage=colorEnhancement(imageDir,name)
    saveImage.save(os.path.join(saveDir,saveName))

