import sys
sys.path.append('../')

from utils.voice.prepare_dataset import WhistleDataloader

def train_model(model_arch, dl_info, **kwargs):
    input_shape = kwargs.get('input_shape', [None, None, 1])
    batch_size = kwargs.get('batch_size', 64)
    initial_epoch = kwargs.get('initial_epoch', 1)
    initial_model_path = kwargs.get('initial_model_path', None)
    num_epochs = kwargs.get('num_epochs', 100)
    dset_reload_freq = kwargs.get('dset_reload_freq', 2)
    train_params = kwargs.get('train_params', None)

    dl_info['batch_size'] = batch_size
    dl_info['epoch_size'] = kwargs.get('epoch_size', 500)

    dataloader = WhistleDataloader(**dl_cfg)

    if not initial_model is None:
        model = tf.keras.models.load_model(initial_model_path)
    else:
        model = model_arch.get_model(input_shape,batch_size)

    for epoch in tqdm(range(num_epochs)):
        if epoch % dset_reload_freq == 0:
            train_dset = dataloader.prepare_epoch_dset()
        model.fit(train_dset,
                  epochs=dset_reload_freq,
                  initial_epoch=epoch)
