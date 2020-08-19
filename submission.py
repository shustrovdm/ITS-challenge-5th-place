import pandas as pd
import pretrainedmodels
from torch import nn
import base64
import torch
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import numpy as np
import ttach as tta
import os

test_data_path = 'test/'
if not os.path.exists('test_masks/'):
        os.mkdir('test_masks/')

path_to_sample_csv = 'sample_submission.csv'
path_to_classification_model = './logs/full_se_resnext50_32x4d_stratified/checkpoints/best_full.pth'
path_to_segmentation_model = './logs_segmentation/best_model_efficientb7_60_250_unet.pth'

class_names = ['c_kefir', 'ent_cloacae', 'klebsiella_pneumoniae', 'moraxella_catarrhalis', 'staphylococcus_aureus', 'staphylococcus_epidermidis']

sample_df = pd.read_csv(path_to_sample_csv, dtype=str)




#Classification model and prepeocess

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

    model = BacteriaModel(model_name, num_classes, pretrained)

    return model


model_name = "se_resnext50_32x4d"
model_cls = get_model(model_name, 6).cuda()
state_dict = torch.load(path_to_classification_model, map_location="cuda:0")
model_cls.load_state_dict(state_dict["model_state_dict"])
model_cls.eval()

classification_preprocessing = compose([
    post_transforms()])



#Segmentation model and preprocess
def to_tensor(x, **kwargs):

    if len(x.shape) == 3:
        return x.transpose(2, 0, 1).astype('float32')
    else:
        return x.astype('float32')


def get_preprocessing():
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [

        albu.Normalize(),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

model_seg = torch.load(path_to_segmentation_model)
#TTA
tta_model = tta.SegmentationTTAWrapper(model_seg, tta.aliases.hflip_transform(), merge_mode='sum')
segmentation_preprocessing = get_preprocessing()

results = []
score = []

for id in sample_df['id']:
    print(id)
    image = cv2.imread(test_data_path + str(id) + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Classification
    sample = classification_preprocessing(image = image)
    image_to_classify = sample['image']

    label = model_cls(image_to_classify.to('cuda:0').unsqueeze(0))
    label = label.argmax(-1).view(-1).cpu()
    print(class_names[label.item()])



    #Segmentation
    sample = segmentation_preprocessing(image=image, mask=image)
    image_to_segment, _ = sample['image'], sample['mask']

    x_tensor = torch.from_numpy(image_to_segment).float().to('cuda:0').unsqueeze(0)



    pr_mask = tta_model(x_tensor)
    pr_mask = torch.sigmoid(pr_mask)
    pr_mask = (pr_mask.squeeze().cpu().detach().numpy() > 0.5).astype(np.uint8)



    mask_from_sample = cv2.imread('sample_masks/' + id + '.png', 0) /255



    sample_contours, sapmle_hierarchy = cv2.findContours(mask_from_sample[:512,:512].astype(np.uint8), cv2.RETR_LIST,  cv2.CHAIN_APPROX_NONE)
    predicted_contours, predicted_hierarchy = cv2.findContours(pr_mask[:512, :512], cv2.RETR_LIST,  cv2.CHAIN_APPROX_NONE)
    final_mask = np.zeros((512,512))


    for sample_contour in sample_contours:
        Flag = False
        for predicted_contour in predicted_contours:
            blank_image_sample = np.zeros((512,512))
            blank_image_predicted = np.zeros((512,512))
            blank_mask = np.zeros((512, 512))
            sample_draw = cv2.fillPoly(blank_image_sample, pts =[sample_contour], color=(255,255,255))
            predicted_draw = cv2.fillPoly(blank_image_predicted, pts =[predicted_contour], color=(255,255,255))
            score_contour = np.count_nonzero(np.logical_and(predicted_draw / 255, sample_draw / 255)) / np.count_nonzero(np.logical_or(predicted_draw / 255, sample_draw / 255))

            if score_contour > 0.01:
                Flag = True
                print('original iou: ', score_contour)


                M_sample = cv2.moments(sample_contour)


                if M_sample["m00"] != 0:
                    cx_sample = int(M_sample['m10'] / M_sample['m00'])
                    cy_sample = int(M_sample['m01'] / M_sample['m00'])
                else:
                    final_mask = cv2.fillPoly(final_mask, pts=[sample_contour], color=(255, 255, 255))
                    continue

                M_predicted = cv2.moments(predicted_contour)
                if M_predicted["m00"] != 0:
                    cx_predicted = int(M_predicted['m10'] / M_predicted['m00'])
                    cy_predicted = int(M_predicted['m01'] / M_predicted['m00'])
                else: continue

                blank_mask = np.zeros((512, 512))

                def contour_position(delta_cx, delta_cy):
                    blank_mask = np.zeros((512, 512))
                    new_sample_contour = sample_contour - [int(delta_cx), int(delta_cy)]
                    blank_mask = cv2.fillPoly(blank_mask, pts=[new_sample_contour],
                                              color=(255, 255, 255))
                    score_result = np.count_nonzero(np.logical_and(predicted_draw / 255, blank_mask / 255)) / np.count_nonzero(
                        np.logical_or(predicted_draw / 255, blank_mask / 255 ))

                    return score_result



                best_score = 0

                for dx in range(-7,7):
                    for dy in range(-7,7):
                        new_score = contour_position(delta_cx=dx, delta_cy=dy)
                        if new_score > best_score:
                            best_score = new_score
                            delta_cx = dx
                            delta_cy = dy




                print('new iou:', best_score)

                if best_score <= score_contour:
                    # continue
                    delta_cx = 0
                    delta_cy = 0





                final_mask = cv2.fillPoly(final_mask, pts=[sample_contour - [delta_cx,delta_cy]], color=(255, 255, 255))



        if Flag == False:
            print('missing countour found.')
            final_mask = cv2.fillPoly(final_mask, pts=[sample_contour], color=(255, 255, 255))

    score += [np.count_nonzero(np.logical_and(final_mask[:512,:512], pr_mask[:512,:512])) /
              (np.count_nonzero(np.logical_or(final_mask[:512,:512], pr_mask[:512,:512])))]

    print('Picture mean iou', [np.count_nonzero(np.logical_and(final_mask[:512,:512], pr_mask[:512,:512])) /
              (np.count_nonzero(np.logical_or(final_mask[:512,:512], pr_mask[:512,:512])))])

    pr_mask[:512,:512] = final_mask[:512,:512] / 255
    cv2.imwrite('test_masks/' + id + '.png', pr_mask * 255)


    with open('test_masks/' + id + '.png', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()







    result = {
        'id': id,
        'class': class_names[label.item()],
        'base64 encoded PNG (mask)': encoded_string
    }

    results.append(result)

print('Final mean iou', np.mean(score))
sub_df = pd.DataFrame(results)
sub_df.to_csv('submission.csv', index=False)
sub_df.head()


