import numpy as np
from scipy.io import loadmat
import pandas as pd

import numpy as np
import os
import cv2


matfile = 'attrann.mat'
imagenet_dir = 'ImageNet/'


def get_scaled_im_tensor(img_ids, target_size, max_size, bboxs, attrs):
    images = []
    attenton_maps = []
    # scales = []
    img_shapes = []
    chosen_idx = []
    max_w = -1
    max_h = -1
    # load each image
    for idx, img_id in enumerate(img_ids):
        im_path = imagenet_dir + img_id + '.jpg'
        try:
            img = cv2.imread(im_path).astype(np.float32)
        except:
            continue
        img_shapes.append([img.shape[0], img.shape[1]]) #(limit_x, limit_y)
        # calculate scale
        old_short = min(img.shape[0: 2])
        old_long = max(img.shape[0: 2])
        new_scale = 1.0 * target_size / old_short
        if old_long * new_scale > max_size:
            new_scale = 1.0 * max_size / old_long
        # scale the image
        img = cv2.resize(img, None, fx = new_scale, fy = new_scale, interpolation = cv2.INTER_LINEAR)
        images.append(img)
        # scales.append([new_scale, new_scale])

        # find the max shape
        if img.shape[0] > max_h:
            max_h = img.shape[0]
        if img.shape[1] > max_w:
            max_w = img.shape[1]
        
        # attention map
        attenton_map = np.zeros((img_shapes[-1][0],img_shapes[-1][1]), dtype=np.uint8)
        bbox = bboxs[idx]
        attenton_map[np.round(img_shapes[-1][0]*bbox[2]).astype(int):np.round(img_shapes[-1][0]*bbox[3]).astype(int), np.round(img_shapes[-1][1]*bbox[0]).astype(int):np.round(img_shapes[-1][1]*bbox[1]).astype(int)] = 1
        attenton_map_scaled = cv2.resize(attenton_map, None, fx = new_scale, fy = new_scale, interpolation = cv2.INTER_NEAREST)
        attenton_maps.append(attenton_map_scaled)

        chosen_idx.append(idx)

    # padding the image to be the max size with 0	
    for idx, img in enumerate(images):
        resize_h = max_h - img.shape[0]
        resize_w = max_w - img.shape[1]
        images[idx] = cv2.copyMakeBorder(img, 0, resize_h, 0, resize_w, 
                                         cv2.BORDER_CONSTANT, value=(0,0,0))[:,:,::-1]
        attention_map = attenton_maps[idx]
        attenton_maps[idx] = cv2.copyMakeBorder(attention_map, 0, resize_h, 0, resize_w, 
                                         cv2.BORDER_CONSTANT, value=(0)).astype(np.bool)

    img_attrs = attrs[chosen_idx] >= 0

    return np.array(images), np.array(attenton_maps), img_attrs, np.array(chosen_idx)


def clean():
    attrann = loadmat(matfile)
    image_names_raw = attrann['attrann'][0][0][0]
    image_names = [image_names_raw[i][0][0] for i in range(len(image_names_raw))]
    flickr_warning = cv2.imread('flickr_warning.jpg')
    count = 1
    dir = os.listdir(imagenet_dir)
    for im in image_names:
        if im + '.jpg' in dir:
            read = cv2.imread(imagenet_dir + im + '.jpg') 
            if read is None or np.sum(read == flickr_warning) == np.prod(flickr_warning.shape):
                os.remove(imagenet_dir + im + '.jpg')
        else:
            count += 1
    exit('not found images count: {}'.format(count)) 


def preview(num_image):
    for im in np.load('image_with_att.npy')[np.random.randint(0, 6765, size=num_image)][:,:,:,::-1]:
        norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 
        cv2.imshow('img', norm_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def preprocess():
    attrann = loadmat(matfile)
    image_names_raw = attrann['attrann'][0][0][0]
    image_names = [image_names_raw[i][0][0] for i in range(len(image_names_raw))]
    image_attrs = attrann['attrann'][0][0][2]
    bboxs_raw = attrann['attrann'][0][0][3]
    bboxs = [[bboxs_raw[j][0][i][0][0] for i in range(4)] for j in range(9600)]
    #[('x1', 'O'), ('x2', 'O'), ('y1', 'O'), ('y2', 'O')]

    target_size = 224
    max_size = 224
    images, attention_maps, attrs = get_scaled_im_tensor(image_names, target_size, max_size, bboxs, image_attrs)

    # storing
    input('Press enter to start storing to npy....')

    print(images.shape)
    np.save('exist_image_idx', images)

    print(attention_maps.shape)
    np.save('att_map', attention_maps)
    np.save('ImageNet_with_att', images * np.expand_dims(attention_maps, -1))

    print(attrs.shape)
    np.save('ImagetNet_attributes', attrs)


if __name__ == "__main__":
    # to_be_removed = [1789, 6034, 6039]
    # for t in ['ImageNet_attributes.npy', 'ImageNet_with_att.npy']:
    #     loaded = np.load(t)
    #     if t == 'ImageNet_attributes.npy':
    #         print(loaded[to_be_removed])
    #     new_t = np.delete(loaded, to_be_removed, 0)
    #     print(new_t.shape)
    #     print(t.replace('.npy', ''))
    #     np.save(t.replace('.npy', ''), new_t)
    pass
    