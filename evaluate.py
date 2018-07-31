import argparse
import json
import os

import cv2 as cv
import keras.backend as K
import numpy as np
from tqdm import tqdm

from config import img_size, image_folder, eval_path, best_model
from model import build_model
from utils import random_crop, preprocess_input, psnr

if __name__ == '__main__':
    names_file = 'valid_names.txt'
    with open(names_file, 'r') as f:
        names = f.read().splitlines()

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scale", help="scale")
    args = vars(ap.parse_args())
    scale = int(args["scale"])

    scale_key = 'x{}'.format(scale)
    model_weights_path = os.path.join('models', best_model[scale_key])
    model = build_model(scale=scale)
    model.load_weights(model_weights_path)

    h, w = img_size * scale, img_size * scale
    psnr_list = []
    total_bicubic = 0

    for i in tqdm(range(len(names))):
        name = names[i]
        filename = os.path.join(image_folder, name)
        image_bgr = cv.imread(filename)
        gt = random_crop(image_bgr, scale)

        input = cv.resize(gt, (img_size, img_size), cv.INTER_CUBIC)
        bicubic = cv.resize(input, (img_size * scale, img_size * scale), cv.INTER_CUBIC)

        x = input.copy()
        x = preprocess_input(x.astype(np.float32))
        x_test = np.empty((1, img_size, img_size, 3), dtype=np.float32)
        x_test[0] = x
        out = model.predict(x_test)
        out = out.reshape((h, w, 3))
        out = np.clip(out, 0.0, 255.0)
        out = out.astype(np.uint8)

        total_bicubic += psnr(bicubic, gt)
        psnr_list.append(psnr(out, gt))

    print('num_valid_samples: ' + str(len(names)))
    print('scale: ' + str(scale))
    print('PSNR(avg): {0:.5f}'.format(np.mean(psnr_list)))
    bicubic_avg = total_bicubic / len(names)
    print('Bicubic(avg): {0:.5f}'.format(bicubic_avg))

    if os.path.isfile(eval_path):
        with open(eval_path) as file:
            eval_result = json.load(file)
    else:
        eval_result = {}
    eval_result['psnr_avg_x{}'.format(scale)] = np.mean(psnr_list)
    eval_result['bicubic_avg_x{}'.format(scale)] = bicubic_avg
    with open(eval_path, 'w') as file:
        json.dump(eval_result, file)

    K.clear_session()
