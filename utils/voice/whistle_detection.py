from tqdm import tqdm
import tensorflow as tf
from .experiment import Experiment

class WhistleDetection(Experiment):
    '''
    '''
    def train(self, num_epochs, validate=True, skip_train=False, init_model=None):
        '''
        '''
        self.logger.log('starting training')

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                    filepath='model.h5',
                                    save_weights_only=True,
                                    monitor='val_accuracy',
                                    mode='max',
                                    save_best_only=True)

        callbacks = [self.logger, model_checkpoint_callback]
        if init_model is not None:
            self.model = tf.keras.models.load_model(init_model)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=self.metrics.metric_set)
        if validate and not skip_train:
            # train and validate model
            self.model.fit_generator(self.dset['train_generator'],
                                epochs=num_epochs,
                                validation_data=self.dset['validation_generator'],
                                callbacks=callbacks)
        elif validate and skip_train:
            # validate/evaluate model
            self.model.evaluate(self.dset['validation_generator'],
                                callbacks=callbacks)
