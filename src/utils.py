from PIL import Image
import numpy as np
import base64
from io import BytesIO

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


def img_encode(img:Image.Image) -> str:
    with BytesIO() as buf:
        img.convert('RGB').save(buf, 'JPEG')
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return encoded

def img_decode(encoded:str) -> Image.Image:
    with BytesIO(base64.b64decode(encoded)) as buf:
        img = Image.open(buf)
        img.load()
    return img

def parse_score(review):
    '''
    Extracts the score from the judgement text. The number that appears last in
    the text is extracted, and if the number is either out of bounds or there is
    no number, 0 is returned.
    '''
    try:
        for word in review.split()[::-1]:
            digs = []
            for c in word[::-1]:
                if not c.isdigit():
                    if len(digs)>0: break
                    else: continue
                else:
                    digs.append(c)
            if len(digs)>0:
                result = int(''.join(digs[::-1]))
                if result>10: return 0
                return result
        return 0
    except ValueError as e:
        return 0



def internvl_build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def internvl_load_image(image, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    transform = internvl_build_transform(input_size=input_size)
    images = internvl_dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def xcomposer_padding_336(b, pad=336):
    width, height = b.size
    tar = int(np.ceil(height / pad) * pad)
    top_padding = 0 # int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = T.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])

    return b

def xcomposer_Image_transform(img, hd_num=25):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width/ height)
    scale = 1
    while scale*np.ceil(scale/ratio) <= hd_num:
        scale += 1
    scale -= 1
    scale = min(np.ceil(width / 560), scale)
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)
    #print (scale, f'{height}/{new_h}, {width}/{new_w}')

    img = T.functional.resize(img, [new_h, new_w],)
    img = xcomposer_padding_336(img, 560)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img

