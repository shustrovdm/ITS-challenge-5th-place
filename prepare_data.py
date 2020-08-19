import glob
import json
import os
from shutil import copyfile
DATA_DIR = './train'

jsons_list = glob.glob(DATA_DIR + '/*.json')

if not os.path.exists('./train_classes_with_json/'):
    os.mkdir('./train_classes_with_json/')

if not os.path.exists('./train_classes/'):
    os.mkdir('./train_classes/')
for jsn in jsons_list:
    with open(jsn, 'r') as data_file:
        json_data = json.load(data_file)

    shapes = json_data['shapes']
    for shape in shapes:
        label = shape['label']

    print(os.path.splitext(os.path.basename(jsn))[0])
    if not os.path.exists('./train_classes_with_json/' + label):
        os.mkdir('./train_classes_with_json/' + label)

    if not os.path.exists('./train_classes/' + label):
        os.mkdir('./train_classes/' + label)
    copyfile(os.path.splitext(jsn)[0] + '.png', './train_classes_with_json/' + label + '/' + os.path.splitext(os.path.basename(jsn))[0] + '.png')
    copyfile(os.path.splitext(jsn)[0] + '.json',
             './train_classes_with_json/' + label + '/' +
             os.path.splitext(os.path.basename(jsn))[0] + '.json')

    copyfile(os.path.splitext(jsn)[0] + '.png',
             './train_classes/' + label + '/' + os.path.splitext(os.path.basename(jsn))[0] + '.png')
