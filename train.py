import argparse

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import patience, epochs, batch_size
from data_generator import train_gen, valid_gen
from model import build_model
from utils import get_example_numbers

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    ap.add_argument("-s", "--scale", help="scale")
    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]
    scale = int(args["scale"])
    checkpoint_models_path = 'models/'

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'model.x' + str(scale) + '-{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
            self.model_to_save.save(fmt % (epoch, logs['val_loss']))


    new_model = build_model(scale=scale)
    if pretrained_path is not None:
        new_model.load_weights(pretrained_path, by_name=True)

    adam = keras.optimizers.Adam(lr=1e-4, epsilon=1e-8, decay=1e-6)
    new_model.compile(optimizer=adam, loss='mean_absolute_error')

    print(new_model.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    num_train_samples, num_valid_samples = get_example_numbers()
    # Start Fine-tuning
    new_model.fit_generator(train_gen(scale=scale),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(scale=scale),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=4
                            )
