import multiprocessing
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import CIRDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def extract_index_features(dataset: CIRDataset, model, device=torch.device('cuda')):
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    classic_val_loader = DataLoader(dataset=dataset, batch_size=64, num_workers=2,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = None
    index_features_0 = None
    index_features_raw = None
    index_features_raw_0 = None

    index_names = []
    if dataset.data_name == 'cirr':
        print(f"extracting CIRR {dataset.split} index features")
    elif dataset.data_name == 'fiq':
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    N = len(dataset)
    cnt = 0
    for names, images in tqdm(classic_val_loader):
        with torch.no_grad():
            images = images.to(device, non_blocking=True)
            if index_features_0 is None and cnt >= N // 2:
                index_features_raw_0 = index_features_raw.to('cpu')
                index_features_raw = None
                index_features_0 = index_features.to('cpu')
                index_features = None
            image_features, image_embeds_frozen = model.blip_model.extract_target_features(images, mode="mean")
            if index_features is None:
                index_features = image_features
                index_features_raw = image_embeds_frozen
            else:
                index_features = torch.cat((index_features, image_features))
                index_features_raw = torch.cat((index_features_raw, image_embeds_frozen))
            index_names.extend(names)
            cnt += images.shape[0]
    index_features_ = index_features.to('cpu')
    index_features_raw_ = index_features_raw.to('cpu')
    index_features_ = torch.cat((index_features_0, index_features_))
    index_features_raw_ = torch.cat((index_features_raw_0, index_features_raw_))
    del index_features_raw
    del index_features
    return (index_features_, index_features_raw_), index_names


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path
    models_path.mkdir(exist_ok=True, parents=True)
    torch.save({
        'epoch': cur_epoch,
        "state_dict": model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))
    print("save model successfully")


def save_checkpoint(name: str, cur_epoch: int,
                    model_to_save: nn.Module,
                    optimizer,
                    scaler,
                    random_state,
                    np_random_state,
                    torch_random_state,
                    torch_cuda_random_state,
                    scheduler,
                    training_path: Path):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path
    models_path.mkdir(exist_ok=True, parents=True)
    torch.save({
        'epoch': cur_epoch,
        "state_dict": model_to_save.state_dict(),
        'optimizer': optimizer,
        'scaler': scaler,
        'random': random_state,
        'np_random': np_random_state,
        'torch_random': torch_random_state,
        'torch_cuda_random': torch_cuda_random_state,
        'scheduler': scheduler
    }, str(models_path / f'{name}.pt'))
    print("save model successfully")


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)
