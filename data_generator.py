import os
import random
from random import shuffle

import cv2 as cv
import imutils
import numpy as np
from keras.utils import Sequence

from config import batch_size, img_size, channel, image_folder
from utils import random_crop, preprocess_input


class DataGenSequence(Sequence):
    def __init__(self, usage, scale):
        self.usage = usage
        self.scale = scale

        if usage == 'train':
            names_file = 'train_names.txt'
        else:
            names_file = 'valid_names.txt'

        with open(names_file, 'r') as f:
            self.names = f.read().splitlines()

        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        out_img_rows, out_img_cols = img_size * self.scale, img_size * self.scale

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_size, img_size, channel), dtype=np.float32)
        batch_y = np.empty((length, out_img_rows, out_img_cols, channel), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i + i_batch]
            filename = os.path.join(image_folder, name)
            # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
            image_bgr = cv.imread(filename)

            gt = random_crop(image_bgr, self.scale)

            if np.random.random_sample() > 0.5:
                gt = np.fliplr(gt)

            angle = random.choice((0, 90, 180, 270))
            gt = imutils.rotate_bound(gt, angle)

            x = cv.resize(gt, (img_size, img_size), cv.INTER_CUBIC)

            batch_x[i_batch, :, :] = preprocess_input(x)
            batch_y[i_batch, :, :] = gt

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen(scale):
    return DataGenSequence('train', scale)


def valid_gen(scale):
    return DataGenSequence('valid', scale)


def split_data():
    names = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    num_samples = len(names)  # 1341430
    print('num_samples: ' + str(num_samples))

    num_train_samples = int(num_samples * 0.992)
    print('num_train_samples: ' + str(num_train_samples))
    num_valid_samples = num_samples - num_train_samples
    print('num_valid_samples: ' + str(num_valid_samples))
    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    shuffle(valid_names)
    shuffle(train_names)

    # with open('names.txt', 'w') as file:
    #     file.write('\n'.join(names))

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == '__main__':
    split_data()
