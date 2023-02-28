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

def saturation(image, level):
    enhancer = ImageEnhance.Color(image)
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

def add_contour(image):
    return image.filter(ImageFilter.CONTOUR)

def add_detail(image):
    return image.filter(ImageFilter.DETAIL)

def add_edge_enhance(image):
    return image.filter(ImageFilter.EDGE_ENHANCE)

def add_emboss(image):
    return image.filter(ImageFilter.EMBOSS)

def add_find_edges(image):
    return image.filter(ImageFilter.FIND_EDGES)

def add_sharpen(image):
    return image.filter(ImageFilter.SHARPEN)

def add_smooth(image):
    return image.filter(ImageFilter.SMOOTH)

def add_vignette(image):
    
    width, height = image.size
    black_area = Image.new('RGB', image.size, (0,0,0))
    mask = Image.new('L', image.size, 255)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=0)
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=10))
    return Image.composite(image, black_area, mask_blurred)

def apply_operation(image, operation_name):
    if operation_name == "Flip":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif operation_name == "Mirror":
        return ImageOps.mirror(image)
    elif operation_name == "Rotate":
        return image.rotate(90)
    else:
        return image


