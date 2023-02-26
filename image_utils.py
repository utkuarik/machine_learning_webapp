import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont

def grayscale(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray_image)

def blur(image):
    blurred_image = image.filter(ImageFilter.BLUR)
    return blurred_image

def brightness(image, level):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(level)

def contrast(image, level):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(level)

def crop(image, width, height):
    w, h = image.size
    left = (w - width) / 2
    top = (h - height) / 2
    right = (w + width) / 2
    bottom = (h + height) / 2
    return image.crop((left, top, right, bottom))

def resize(image, width, height):
    return image.resize((width, height))

def sepia(image):
    sepia_filter = ImageFilter.Kernel((3,3), (0.393,0.769,0.189,0.349,0.686,0.168,0.272,0.534,0.131))
    sepia_image = image.filter(sepia_filter)
    return sepia_image

def add_text(image, text, font_size, color):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text((10, 10), text, color, font=font)
    return image

def add_sticker(image, sticker):
    sticker = Image.open(sticker)
    image.paste(sticker, (50, 50), sticker)
    return image

