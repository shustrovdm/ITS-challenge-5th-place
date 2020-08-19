import base64
import cv2
import numpy as np
import os
import pandas as pd

df = pd.read_csv('sample_submission.csv')

if not os.path.exists('sample_masks/'):
        os.mkdir('sample_masks/')

for i,id in enumerate(df['id']):
    # print(str(name))
    name = str(id)
    if len(str(df.at[i, 'id'])) < 2:
        name = '00' + name
    elif len(str(df.at[i, 'id'])) < 3:
        name = '0' + name
    with open('sample_masks/' + name + '.png', 'wb') as fp:

        fp.write(base64.b64decode(df.at[i, 'base64 encoded PNG (mask)'].encode()))

    # print(str(df['id'][i]))


    print(name)
    image = cv2.imread('sample_masks/' + name + '.png', 0)
    image = cv2.rotate(image, rotateCode = cv2.ROTATE_90_CLOCKWISE)
    image = cv2.flip(image,1)
    blank = np.zeros((512,640))
    blank[:512,:512] = image[:512,:]
    # image = image[:512,:]
    # print(image.shape)
    cv2.imwrite('sample_masks/' + name + '.png', blank)

