import os
import json
from torch.utils.data import Dataset as BaseDataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from pytorch_toolbelt.inference.tiles import ImageSlicer
import albumentations as albu
import cv2
import glob
import random
from PIL import Image, ImageDraw
import torch
from pytorch_toolbelt.utils import fs
import numpy as np
from torch.utils.data import WeightedRandomSampler, Dataset, ConcatDataset

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        #
        albu.VerticalFlip(p=0.5),
        # albu.RandomRotate90(p=0.5),
        # albu.PadIfNeeded(min_height=512, min_width=640, always_apply=True, border_mode=0),
        # albu.IAAFliplr(p=0.5),
        # albu.IAAFlipud(p=0.5),
        # albu.augmentations.transforms.MaskDropout(max_objects=5,p=0.6),

        # albu.Flip(p=0.3),
        # albu.ToSepia(p = 0.5),
        # albu.ToGray(p= 1.),

        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=40, shift_limit=0.1, p=0.5, border_mode=2),
        # albu.Rotate(p=0.5),
        # albu.ChannelShuffle(p=0.5),
        # #
        # albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=512, width=512, p=0.5),
        # # albu.CropNonEmptyMaskIfExists(height=512, width=640, p=0.5),
        albu.augmentations.transforms.MaskDropout(max_objects=1, p=0.4),
        albu.augmentations.transforms.GridDropout(unit_size_min = 60, unit_size_max= 250,  fill_value=0, mask_fill_value=0,
                                                  holes_number_x = None, holes_number_y = None, random_offset = False,  p= 1.0),

        # #
        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),
        # #
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
                albu.RGBShift(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
        # aug(image = 'image')

        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         # albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.6,
        # ),

        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
        albu.Normalize(),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(256, 256),
        # albu.ToGray(p=1.),
        albu.Normalize()
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):

    if len(x.shape) == 3:
        return x.transpose(2, 0, 1).astype('float32')
        # print(x.shape)
        # return x.astype('float32')
    else:
        # return torch.tensor(x.astype('float32'))
        return torch.tensor(x.astype('float32')).unsqueeze(0)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        # albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



def polygons_to_mask_array(polygons, width: int = 384, height: int = 384) -> np.ndarray:
    '''
    This function takes a list of lists that contains polygon masks for each building. Example;

    [[x11,y11,x12,y12,...],...,[xn1,yn1,xn2,yn2,...]]

    The return of this function is an array of size width x height which contains a binary mask
    as defined by the list of polygons. This will be the target for our network!
    '''

    img = Image.new('L', (width, height), 0)
    for polygon in polygons:
        nested_lst_of_tuples = [tuple(l) for l in polygon['points']]
        ImageDraw.Draw(img).polygon(nested_lst_of_tuples, outline=1, fill=1)
    mask = np.array(img)

    return mask



class FullSizedDataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['bacteria']

    def __init__(
            self,
            directory,
            training = False,
            classes=None,
            augmentation=None,
            preprocessing=None,
            ids = None,
    ):
        self.ids = ids
        # self.jsons = [os.path.join(directory, id) for id in self.ids]
        self.jsons = self.ids
        # self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        # self.slicer = ImageSlicer(image_shape =(512,640),tile_size =(128,128), tile_step = 0, image_margin = 0)


        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.training = training

    def __getitem__(self, i):

        # read data
        with open(self.jsons[i], 'r') as data_file:
            json_data = json.load(data_file)

        # print(json_data["imagePath"])
        image = cv2.imread(os.path.splitext(self.jsons[i])[0] + '.png')
        # image = cv2.resize(image, (256,256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        mask = polygons_to_mask_array(json_data['shapes'], 640, 512)
        # mask = cv2.resize(mask, (256, 256))
        # mask = cv2.resize(mask, (384,384))
        # mask = cv2.imread(self.masks_fps[i], 0)


        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # if self.training:
        #     # print(image.shape)
        #     aug = imgaug.augmenters.geometric.Jigsaw(nb_rows=16, nb_cols=16, max_steps=1, allow_pad=False)
        #     mask = SegmentationMapsOnImage(mask, shape = mask.shape)
        #     image, mask = aug(image=image, segmentation_maps=mask)
        #     mask = mask.get_arr()



        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # image = image / 255
        return image, mask

    def __len__(self):
        return len(self.ids)




def get_train_val_datasets(directory, seed = 789, preprocessing_fn = None):
    classes = glob.glob(directory + '/*')
    train_ids = []
    valid_ids = []
    # c_kefir ent_cloacae klebsiella_pneumoniae moraxella_catarrhalis staphylococcus_aureus staphylococcus_epidermidis
    for cls in classes:
        print(cls)
        # if 'c_kefir' in cls or 'ent_cloacae' in cls or 'klebsiella_pneumoniae' in cls:
        #     continue
        # print(cls)
        json_ids = glob.glob(cls + '/*.json')
        # random.Random()
        random.Random(seed).shuffle(json_ids)
        train_ids.append(json_ids[:int((len(json_ids)+1)*.85)])
        valid_ids.append(json_ids[int((len(json_ids)+1)*.85):])
    train_ids = [item for sublist in train_ids for item in sublist]
    valid_ids = [item for sublist in valid_ids for item in sublist]
    print(valid_ids)
    print('Images in train: ', len(train_ids))
    print('Images in valid: ', len(valid_ids))

    train_dataset = FullSizedDataset(directory,augmentation=get_training_augmentation(),
                            preprocessing=get_preprocessing(preprocessing_fn),
                            classes=['bacteria'], ids = train_ids, training=True)

    valid_dataset = FullSizedDataset(directory,augmentation= get_validation_augmentation(),
                            preprocessing=get_preprocessing(preprocessing_fn),
                            classes=['bacteria'], ids = valid_ids)
    return train_dataset, valid_dataset


