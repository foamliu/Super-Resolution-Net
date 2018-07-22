import json
import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

from config import img_size, image_folder, scale, eval_path
from model import build_model
from utils import random_crop, preprocess_input, psnr

if __name__ == '__main__':
    model_weights_path = 'models/model.16-9.0500.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    names_file = 'valid_names.txt'
    with open(names_file, 'r') as f:
        names = f.read().splitlines()

    h, w = img_size * scale, img_size * scale
    psnr_list = []
    total_bicubic_x4 = 0

    for i in tqdm(range(len(names))):
        name = names[i]
        filename = os.path.join(image_folder, name)
        image_bgr = cv.imread(filename)
        gt = random_crop(image_bgr)

        input = cv.resize(gt, (img_size, img_size), cv.INTER_CUBIC)
        bicubic_x4 = cv.resize(input, (img_size * 4, img_size * 4), cv.INTER_CUBIC)

        x = input.copy()
        x = preprocess_input(x.astype(np.float32))
        x_test = np.empty((1, img_size, img_size, 3), dtype=np.float32)
        x_test[0] = x
        out = model.predict(x_test)
        out = out.reshape((h, w, 3))
        out = np.clip(out, 0.0, 255.0)
        out = out.astype(np.uint8)

        total_bicubic_x4 = psnr(bicubic_x4, gt)
        psnr_list.append(psnr(out, gt))

    print('num_valid_samples: ' + str(len(names)))
    print('PSNR(avg): {0:.5f}'.format(np.mean(psnr_list)))
    bicubic_avg_x4 = total_bicubic_x4 / len(names)
    print('Bicubicx4(avg): {0:.5f}'.format(bicubic_avg_x4))

    if os.path.isfile(eval_path):
        with open(eval_path) as file:
            eval_result = json.load(file)
    else:
        eval_result = {}
    eval_result['psnr_avg'] = np.mean(psnr_list)
    eval_result['bicubic_avg_x4'] = np.mean(bicubic_avg_x4)
    with open(eval_path, 'w') as file:
        json.dump(eval_result, file)
