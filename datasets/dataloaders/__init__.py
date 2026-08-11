from datasets.dataloaders.detection import get_dataloaders_detection
from datasets.dataloaders.train import get_dataloaders_train
from datasets.dataloaders.train_disk import get_dataloaders_train_disk

def get_dataloaders(name, **kwargs):
    if name == "train":
        return get_dataloaders_train(**kwargs)
    elif name == "detection":
        return get_dataloaders_detection(**kwargs)
    elif name == "train_disk":
        return get_dataloaders_train_disk(**kwargs)
    else:
        raise NotImplementedError
