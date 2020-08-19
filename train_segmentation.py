import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from pytorch_toolbelt import losses as L

if not os.path.exists('logs_segmentation/'):
        os.mkdir('logs_segmentation/')

#c_kefir ent_cloacae kleiella_pneumoniae moraxella_catarrhalis staphylococcus_aureus staphylococcus_epidermidis
Experiment = 'efficientb7_60_250_unet'

DATA_DIR = 'train_classes_with_json/'


from catalyst.utils import set_global_seed

set_global_seed(345)
torch.manual_seed(345)
torch.cuda.manual_seed_all(345)
np.random.seed(345)
torch.cuda.manual_seed(345)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import imgaug
from dataset import *


def _init_fn(worker_id):
    np.random.seed(int(345))

ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['bacteria']
ACTIVATION = None  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset, valid_dataset = get_train_val_datasets(DATA_DIR, seed = 345, preprocessing_fn = preprocessing_fn)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, worker_init_fn=_init_fn, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=0, worker_init_fn=_init_fn)

#MODEL
def get_seg_model(ENCODER = ENCODER,
    ENCODER_WEIGHTS = ENCODER_WEIGHTS,
    ACTIVATION = ACTIVATION,
                  ):
    CLASSES = ['bacteria']

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        # decoder_use_batchnorm = True,
        # decoder_attention_type = 'scse',
        # decoder_merge_policy = 'cat',
        # decoder_dropout=0.2,
        # decoder_segmentation_channels = 128,
        # decoder_segmentation_channels=128,
        # encoder_depth = 7,
        # decoder_channels = [1024, 512, 256, 128, 64, 32, 16],
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    model = nn.DataParallel(model)
    return model


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    __name__ = 'joint_loss'

    def __init__(self, first, second, third, first_weight=1.0, second_weight=1.0, third_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)
        self.third = WeightedLoss(third, third_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input) + self.third(*input)


model = get_seg_model()

loss_1 = smp.utils.losses.BCEWithLogitsLoss()

loss_2 = torch.nn.BCELoss()

loss_3 = L.BinaryFocalLoss()

loss_4 = L.BinaryLovaszLoss()

loss_5 = L.DiceLoss(mode = 'binary')

loss_6 = L.SoftBCEWithLogitsLoss()

loss_7 = L.JaccardLoss(mode = 'binary')


loss = JointLoss(loss_3,loss_4,loss_3, 0.0, 0.7, 0.3)
metrics = [
    smp.utils.metrics.IoU(threshold=0.5, activation='sigmoid'),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=5e-4),
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 15, threshold=1e-6)


train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

max_score = 0


parents = model.children()


# print(parents)
for i in range(0, 120):


    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    torch.save(model, 'logs_segmentation/last' + Experiment + '.pth')
    valid_logs = valid_epoch.run(valid_loader)
    optimizer.step()
    scheduler.step(valid_logs['iou_score'])
    # scheduler.step()
    # print(scheduler.get_lr())
    with torch.no_grad():
        model_last = torch.load('logs_segmentation/last' + Experiment + '.pth')
        model_last = model_last
        # tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.hflip_transform(), merge_mode='sum')



        score = []
        score_tta = []
        score_06 = []
        score_04 = []
        model_last.eval()
        # tta_model.eval()
        for n in range(valid_dataset.__len__()):

            image, gt_mask = valid_dataset[n]
        # image, gt_mask = image, gt_mask.transpose(1, 2, 0)
            gt_mask = gt_mask.squeeze()

            x_tensor = torch.from_numpy(image).float().to(DEVICE).unsqueeze(0)
            # pr_mask_tta = tta_model(x_tensor)
            # pr_mask_tta = torch.sigmoid(pr_mask_tta)
            pr_mask = model_last(x_tensor)
            pr_mask = torch.sigmoid(pr_mask)

        # score_06 += [np.count_nonzero(np.logical_and(gt_mask, (pr_mask.squeeze().cpu().numpy() > 0.6).astype(np.uint8))) /
        #             np.count_nonzero(np.logical_or(gt_mask, (pr_mask.squeeze().cpu().numpy() > 0.6).astype(np.uint8)))]
        # score_04 += [np.count_nonzero(np.logical_and(gt_mask, (pr_mask.squeeze().cpu().numpy() > 0.4).astype(np.uint8))) /
        #             np.count_nonzero(np.logical_or(gt_mask, (pr_mask.squeeze().cpu().numpy() > 0.4).astype(np.uint8)))]
        #     pr_mask_tta = (pr_mask_tta.squeeze().cpu().detach().numpy() > 0.5).astype(np.uint8)
            pr_mask = (pr_mask.squeeze().cpu().detach().numpy() > 0.5).astype(np.uint8)
        # cv2.imwrite()
        # pr_mask = cv2.resize(pr_mask, (512,640))
        # gt_mask = cv2.resize(gt_mask.squeeze().cpu().numpy(), (512,640))

            # score_tta += [np.count_nonzero(np.logical_and(gt_mask, pr_mask_tta)) /
            #            (np.count_nonzero(np.logical_or(gt_mask, pr_mask_tta)) + 1e-7)]

            score += [np.count_nonzero(np.logical_and(gt_mask, pr_mask)) /
                    (np.count_nonzero(np.logical_or(gt_mask, pr_mask))+ 1e-7)]


    # print(valid_logs)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print('Thrs 0.5 Valid iou:', np.mean(score))
        # print('Thrs 0.5 Valid TTA iou:', np.mean(score_tta))
    # print('Thrs 0.6 Valid iou:', np.mean(score_06))
    # print('Thrs 0.4 Valid iou:', np.mean(score_04))




    # do something (save model, change lr, etc.)
    if max_score < np.mean(score):
        # print(valid_logs)
        max_score = np.mean(score)

        torch.save(model, 'logs_segmentation/best_model_' + Experiment + '.pth')
        print('New Best Model saved! score:', max_score)


            # pr_mask = cv2.resize(pr_mask, (512,640))
            # gt_mask = cv2.resize(gt_mask.squeeze().cpu().numpy(), (512,640))


    # if i % 10 == 0:
    #     torch.save(model, 'logs_segmentation/model_' + str(i) +'.pth')
    #     print('Model saved!')
        # best_model = torch.load('./best_model.pth')
        # n = np.random.choice(len(train_dataset))
        #
        # image, gt_mask = train_dataset[n]
        #
        # gt_mask = gt_mask.squeeze()
        #
        # x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # pr_mask = best_model.predict(x_tensor)
        # pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        # cv2.imwrite('prediction.png', pr_mask*255)



        # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
        # print('Decrease decoder learning rate to ', optimizer.param_groups[0]['lr'])
    # if i == 30:
    #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    #     print('Decrease decoder learning rate to 1e-6!')


