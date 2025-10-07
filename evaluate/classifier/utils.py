
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .weight_averaging import EMA



def get_attribute_ids(attribute_size):
    attribute_indices = {}
    idx = 0
    for attr, size in attribute_size.items():
        attribute_indices[attr] = list(range(idx, idx + size))
        idx += size
    return attribute_indices

def generate_checkpoint_callback(model_name, dir_path, monitor="val_loss", mode="min", save_last=False, top=1):
    checkpoint_callback = ModelCheckpoint(
    dirpath=dir_path,
    filename= model_name + '-{epoch:02d}',
    monitor=monitor,  # Disable monitoring for checkpoint saving,
    mode = mode,
    save_top_k=top,
    save_last=save_last
    )
    return checkpoint_callback

def generate_early_stopping_callback(patience=5, min_delta = 0.001, monitor="val_loss", mode="min"):
    early_stopping_callback = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, mode = mode)
    return early_stopping_callback

def generate_ema_callback(decay=0.999):
    ema_callback=EMA(decay=decay)
    return ema_callback
