img_size = 48
channel = 3
kernel = 3
batch_size = 16
epochs = 10000
patience = 50
# num_train_samples = 519244
# num_valid_samples = 4188
epsilon = 1e-6
epsilon_sqr = epsilon ** 2
image_folder = '/mnt/code/ImageNet-Downloader/image/resized'
eval_path = 'eval.json'

best_model = {'x2': 'model.x2-03-3.9841.hdf5', 'x3': 'model.x3-10-6.6455.hdf5', 'x4': 'model.x4-09-7.9866.hdf5'}

