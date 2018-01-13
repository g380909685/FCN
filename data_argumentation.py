import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import PIL.Image as Image


img_root = '/home/zeng/data/datasets/segmentation_Dataset/VOC2012/JPEGImages'
map_root = '/home/zeng/data/datasets/segmentation_Dataset/VOC2012/SegmentationClass'
output_root_map = '/home/zeng/data/datasets/segmentation_Dataset/VOC2012/ASegmentationClass'
output_root_img = '/home/zeng/data/datasets/segmentation_Dataset/VOC2012/AJPEGImages'

files = os.listdir(map_root)
it = 1
for img_name in files:
    if not img_name.endswith('.png'):
        continue
    pass
    print it
    it += 1
    img = Image.open(os.path.join(img_root, img_name[:-4])+".jpg")
    map = Image.open(os.path.join(map_root, img_name[:-4])+".png")
    r, c = img.size
    half_remove_r = np.ceil(r/20).astype('int')
    half_remove_c = np.ceil(c/20).astype('int')

    img_l = img.crop((0, 0, c-2*half_remove_c,r))
    img_t = img.crop((0, 0, c, r-2*half_remove_r))
    img_r = img.crop((2*half_remove_c, 0, c, r))
    img_b = img.crop((0, 2*half_remove_r, c, r))
    img_c = img.crop((half_remove_c, half_remove_r, c-half_remove_c, r-half_remove_r))
    img_l.save(os.path.join(output_root_img, img_name[:-4]+"_left"), "JPEG")
    img_t.save(os.path.join(output_root_img, img_name[:-4]+"_top"), "JPEG")
    img_r.save(os.path.join(output_root_img, img_name[:-4]+"_right"), "JPEG")
    img_b.save(os.path.join(output_root_img, img_name[:-4]+"_bottom"), "JPEG")
    img_c.save(os.path.join(output_root_img, img_name[:-4]+"_center"), "JPEG")

    map_l = map.crop((0, 0, c-2*half_remove_c,r))
    map_t = map.crop((0, 0, c, r-2*half_remove_r))
    map_r = map.crop((2*half_remove_c, 0, c, r))
    map_b = map.crop((0, 2*half_remove_r, c, r))
    map_c = map.crop((half_remove_c, half_remove_r, c-half_remove_c, r-half_remove_r))
    map_l.save(os.path.join(output_root_map, img_name[:-4]+"_left"), "PNG")
    map_t.save(os.path.join(output_root_map, img_name[:-4]+"_top"), "PNG")
    map_r.save(os.path.join(output_root_map, img_name[:-4]+"_right"), "PNG")
    map_b.save(os.path.join(output_root_map, img_name[:-4]+"_bottom"), "PNG")
    map_c.save(os.path.join(output_root_map, img_name[:-4]+"_center"), "PNG")


