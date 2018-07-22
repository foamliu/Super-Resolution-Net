# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from config import img_size, scale, eval_path
from model import build_model
from utils import random_crop, preprocess_input, psnr

if __name__ == '__main__':
    model_weights_path = 'models/model.16-9.0500.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    print(model.summary())

    image_folder = '/mnt/code/ImageNet-Downloader/image/resized'
    names_file = 'valid_names.txt'
    with open(names_file, 'r') as f:
        names = f.read().splitlines()

    samples = random.sample(names, 10)
    h, w = img_size * scale, img_size * scale
    psnr_list = []

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        image_bgr = cv.imread(filename)
        gt = random_crop(image_bgr)

        x = cv.resize(gt, (img_size, img_size), cv.INTER_CUBIC)
        image = cv.resize(x, (h, w), cv.INTER_CUBIC)

        x = preprocess_input(x.astype(np.float32))
        x_test = np.empty((1, img_size, img_size, 3), dtype=np.float32)
        x_test[0] = x
        out = model.predict(x_test)
        out = out.reshape((h, w, 3))
        out = np.clip(out, 0.0, 255.0)
        out = out.astype(np.uint8)

        psnr_list.append(psnr(out, gt))

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_image.png'.format(i), image)
        cv.imwrite('images/{}_gt.png'.format(i), gt)
        cv.imwrite('images/{}_out.png'.format(i), out)

    if os.path.isfile(eval_path):
        with open(eval_path) as file:
            eval_result = json.load(file)
    else:
        eval_result = {}
    eval_result['psnr_list'] = psnr_list
    with open(eval_path, 'w') as file:
        json.dump(eval_result, file, indent=4)

    K.clear_session()
