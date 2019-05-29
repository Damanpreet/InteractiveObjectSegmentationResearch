import os
import os.path as osp
import numpy as np
import argparse
import cv2
import scipy.misc as misc

IMAGE_PATH = './dataset/VOCdevkit/VOC2012/semantic_results'
VAL_LIST = './dataset/VOCdevkit/VOC2012/sample_PN_converted/val_converted_pascal.txt'
TRAIN_LIST = './dataset/VOCdevkit/VOC2012/sample_PN_converted/train_converted_pascal.txt'
FILE_EXT = '_SEG.tif'

def transform(file_name, flipH=False, file_type=None, flag='N'):
    image_options = {"resize":True, "resize_size":[321,321]}

    if file_type=='tiff':
        image = tiff_imread(file_name)
        print(image.shape)
        if(len(image.shape) == 3 and flag=='Y'):
            image = np.transpose(image, [1, 2, 0])
    elif file_type == 'grey':
        image = misc.imread(file_name, mode='P')
    else:
        image = cv2.imread(file_name)

    # resize if needed
    if image_options.get("resize", False) and image_options["resize_size"]:
        ht, wd = image_options["resize_size"]
        interp = cv2.INTER_LINEAR if len(image.shape) > 2 else cv2.INTER_NEAREST
        resized_image = cv2.resize(image, (ht, wd), interpolation=interp)
    else:
        resized_image = image

    # do Horizontal flip for data augmentation
    if flipH:
        flip_image = np.fliplr(resized_image)
    else:
        flip_image = resized_image


    return np.asarray(flip_image)


if __name__=='__main__':
    
    image_list = []

    # Arguments
    parser = argparse.ArgumentParser()  
    parser.add_argument('--image_path', type=str, default=IMAGE_PATH, help='Image Directory')
    parser.add_argument('--val_list', type=str, default=VAL_LIST, help='Validation dataset list')
    parser.add_argument('--train_list', type=str, default=TRAIN_LIST, help='Validation dataset directory')
    parser.add_argument('--file_ext', type=str, default=FILE_EXT, help='File extension')
    args = parser.parse_args()
   
    f = open(args.train_list,'r')
    for line in f:
        try:
            img_name, sample_name = line.strip("\n").split(' ')
        except ValueError:
            img_name = sample_name = line.strip("\n")

        image_list.append(osp.join(args.image_path, img_name+args.file_ext))
        

    images = np.array([transform(image_list[k], False, 'tiff') for k in range(len(image_list))])
    
    img_mean = np.sum(images, axis = (0, 1, 2))/(images.shape[0]*images.shape[1]*images.shape[2])
    print('\nMean of image: ', img_mean)
