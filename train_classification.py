from typing import Callable

import torch
import catalyst
from catalyst.data import BalanceClassSampler
from catalyst.contrib.data.cv import ImageReader
import collections
from catalyst.data import Augmentor
from catalyst.data import ScalarReader, ReaderCompose
from catalyst.dl.callbacks import AccuracyCallback, F1ScoreCallback, ConfusionMatrixCallback
from torch import nn
import pretrainedmodels
import numpy as np
from catalyst.utils import split_dataframe_train_test
from catalyst.dl import SupervisedRunner, utils
from catalyst.utils import (
    create_dataset, create_dataframe, get_dataset_labeling, map_dataframe
)
from catalyst.data import ListDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "" - CPU, "0" - 1 GPU, "0,1" - MultiGPU

logdir = "./logs/full_se_resnext50_32x4d_stratified"
print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")
import random

SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)



ROOT = 'train_classes/'

ALL_IMAGES = list(Path(ROOT).glob("**/*.*"))
ALL_IMAGES = list(filter(lambda x: not x.name.startswith("."), ALL_IMAGES))
print("Number of images:", len(ALL_IMAGES))

dataset = create_dataset(dirs=f"{ROOT}/*", extension="*.png")

df = create_dataframe(dataset, columns=["class", "filepath"])
tag_to_label = get_dataset_labeling(df, "class")

class_names = [
    name for name, id_ in sorted(tag_to_label.items(), key=lambda x: x[1])
]
print(class_names)
df_with_labels = map_dataframe(
    df,
    tag_column="class",
    class_column="label",
    tag2class=tag_to_label,
    verbose=False
)

train_data, valid_data = split_dataframe_train_test(
    df_with_labels, test_size=0.2, random_state=SEED)
train_data, valid_data = (
    train_data.to_dict('records'), valid_data.to_dict('records')
)
# full_data = df_with_labels.to_dict('records')
num_classes = len(tag_to_label)
print(num_classes)

open_fn = ReaderCompose([

    # Reads images from the `rootpath` folder
    # using the key `input_key =" filepath "` (here should be the filename)
    # and writes it to the output dictionary by `output_key="features"` key
    ImageReader(
        input_key="filepath",
        output_key="features",
        rootpath=ROOT
    ),

    # Reads a number from our dataframe
    # by the key `input_key =" label "` to np.long
    # and writes it to the output dictionary by `output_key="targets"` key
    ScalarReader(
        input_key="label",
        output_key="targets",
        default_value=-1,
        dtype=np.int64
    ),

    # Same as above, but with one encoding
    ScalarReader(
        input_key="label",
        output_key="targets_one_hot",
        default_value=-1,
        dtype=np.int64,
        one_hot_classes=num_classes
    )
])



BORDER_CONSTANT = 0
BORDER_REFLECT = 2


def pre_transforms(image_size=512):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT)
    ]

    return result



def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensorV2()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result

train_transforms = compose([
    # pre_transforms(),
    # hard_transforms(),
    post_transforms()
])

valid_transforms = compose([post_transforms()])
    # [pre_transforms(),


# Takes an image from the input dictionary by the key `dict_key`
# and performs `train_transforms` on it.
train_data_transforms = Augmentor(
    dict_key="features",
    augment_fn=lambda x: train_transforms(image=x)["image"]
)


# Similarly for the validation part of the dataset.
# we only perform squaring, normalization and ToTensor
valid_data_transforms = Augmentor(
    dict_key="features",
    augment_fn=lambda x: valid_transforms(image=x)["image"]
)

labels = [x["label"] for x in train_data]
sampler = BalanceClassSampler(labels, mode="upsampling")

def get_loaders(
        open_fn: Callable,
        train_transforms_fn,
        valid_transforms_fn,
        batch_size: int = 32,
        num_workers: int = 4,
        sampler=None
        ) -> collections.OrderedDict:
    """
    Args:
        open_fn: Reader for reading data from a dataframe
        train_transforms_fn: Augmentor for train part
        valid_transforms_fn: Augmentor for valid part
        batch_size: batch size
        num_workers: How many subprocesses to use to load data,
        sampler: An object of the torch.utils.data.Sampler class
            for the dataset data sampling strategy specification
    """
    train_loader = utils.get_loader(
        train_data,
        open_fn=open_fn,
        dict_transform=train_transforms_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=sampler is None,  # shuffle data only if Sampler is not specified (PyTorch requirement)
        drop_last=False,
    )




    valid_loader = utils.get_loader(
        valid_data,
        open_fn=open_fn,
        dict_transform=valid_transforms_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=None,
        drop_last=False,
    )




    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders

loaders = get_loaders(
    open_fn=open_fn,
    train_transforms_fn=train_data_transforms,
    valid_transforms_fn=valid_data_transforms,
    batch_size=8,
)

class BacteriaModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: str = "imagenet"):
        super(BacteriaModel, self).__init__()

        self.model_fn = pretrainedmodels.__dict__[model_name]
        self.model = self.model_fn(num_classes=1000, pretrained=pretrained)
        self._dropout = torch.nn.Dropout(0.2)
        self._avg_pooling = torch.nn.AdaptiveAvgPool2d(1)

        # self.model.fc = nn.Sequential()
        self.dim_feats = self.model.last_linear.in_features
        self.linear = nn.Linear(self.dim_feats, num_classes)



    def forward(self, x):
        x = self.model.features(x)
        x = self._avg_pooling(x)

        features = self._dropout(x)

        batch_size, channels, height, width = features.shape
        features = features.view(batch_size, channels * height * width)
        bacteria_class = self.linear(features)

        return bacteria_class


def get_model(model_name: str, num_classes: int, pretrained: str = "imagenet"):
    # model_fn = pretrainedmodels.__dict__[model_name]
    # model = model_fn(num_classes=1000, pretrained=pretrained)
    # x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
    #
    # model.fc = nn.Sequential()
    # dim_feats = model.last_linear.in_features
    # model.last_linear = nn.Linear(dim_feats, num_classes)

    model = BacteriaModel(model_name, num_classes, pretrained)

    return model

model_name = "se_resnext50_32x4d"
model = get_model(model_name, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[9], gamma=0.3
)

num_epochs = 50


device = utils.get_device()
print(f"device: {device}")

runner = SupervisedRunner(device=device)

callbacks = [
    AccuracyCallback(num_classes=num_classes, activation="Softmax"),

    F1ScoreCallback(
        input_key="targets_one_hot",
        activation="Softmax"
    ),
    ConfusionMatrixCallback(num_classes=num_classes, class_names=class_names)
]



runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    # scheduler=scheduler,
    # our dataloaders
    loaders=loaders,
    # We can specify the callbacks list for the experiment;
    # For this task, we will check accuracy, AUC and F1 metrics
    callbacks=callbacks,
    # path to save logs
    logdir=logdir,
    num_epochs=num_epochs,
    # save our best checkpoint by AUC metric
    main_metric="f1_score",
    # AUC needs to be maximized.
    minimize_metric=False,
    # for FP16. It uses the variable from the very first cell
    # prints train logs
    # monitoring_params = monitoring_params,
    verbose=True,
)