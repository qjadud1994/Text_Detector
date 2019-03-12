'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import cv2
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop

class ListDataset(data.Dataset):
    def __init__(self, root, dataset, train, transform, input_size, multi_scale=False):
        '''
        Args:
          root: (str) DB root ditectory.
          dataset: (str) Dataset name(dir).
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
          multi_scale: (bool) use multi-scale training or not.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.multi_scale = multi_scale
        self.MULTI_SCALES = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960]  #step1, 2
        #self.MULTI_SCALES = [960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280] #step3

        self.encoder = DataEncoder()

        if "SynthText" in dataset:
            self.get_SynthText()
        if "ICDAR2015" in dataset:
            self.get_ICDAR2015()
        if "MLT" in dataset:
            self.get_MLT()
        if "ICDAR2013" in dataset:
            self.get_ICDAR2013()
            
    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) dataset index.

        Returns:
          image: (tensor) image array.
          boxes: (tensor) boxes array.
          labels: (tensor) labels array.
        '''
        # Load image, boxes and labels.
        fname = self.fnames[idx]

        img = cv2.imread(os.path.join(self.root, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = self.boxes[idx].copy()
        labels = self.labels[idx]

        return {"image" : img, "boxes" : boxes, "labels" : labels}

    def collate_fn(self, batch):
        '''bbox encode and make batch

        Args:
          batch: (dict list) images, boxes and labels

        Returns:
          batch_images, batch_loc, batch_cls
        '''
        size = self.input_size
        if self.multi_scale: # get random input_size for multi-scale traininig
            random_choice = random.randint(0, len(self.MULTI_SCALES)-1)
            size = self.MULTI_SCALES[random_choice]

        inputs = torch.zeros(len(batch), 3, size, size)
        loc_targets = []
        cls_targets = []

        for n, data in enumerate(batch):
            img, boxes, labels = self.transform(size=size)(data['image'], data['boxes'], data['labels'])
            inputs[n] = img
            loc_target, cls_target = self.encoder.encode(boxes, labels, input_size=(size, size))
            
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples

    def get_SynthText(self):
        import scipy.io as sio
        data_dir = os.path.join(self.root, 'SynthText/train/')

        gt = sio.loadmat(data_dir + 'gt.mat')
        dataset_size = gt['imnames'].shape[1]
        img_files = gt['imnames'][0]
        labels = gt['wordBB'][0]

        self.num_samples = dataset_size
        print("Training on SynthText : ", dataset_size)

        for i in range(dataset_size):
            img_file = data_dir + str(img_files[i][0])
            label = labels[i]

            _quad = []
            _classes = []

            if label.ndim == 3:
                for i in range(label.shape[2]):
                    _x0 = label[0][0][i]
                    _y0 = label[1][0][i]
                    _x1 = label[0][1][i]
                    _y1 = label[1][1][i]
                    _x2 = label[0][2][i]
                    _y2 = label[1][2][i]
                    _x3 = label[0][3][i]
                    _y3 = label[1][3][i]

                    _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                    _classes.append(1)

            else:
                _x0 = label[0][0]
                _y0 = label[1][0]
                _x1 = label[0][1]
                _y1 = label[1][1]
                _x2 = label[0][2]
                _y2 = label[1][2]
                _x3 = label[0][3]
                _y3 = label[1][3]

                _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                _classes.append(1)

            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))

    def get_ICDAR2015(self):
        data_dir = os.path.join(self.root, 'ICDAR2015_Incidental/')

        dataset_list = os.listdir(data_dir + "train")
        dataset_list = [l[:-4] for l in dataset_list if "jpg" in l]

        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on ICDAR2015 : ", dataset_size)

        for i in dataset_list:
            img_file = data_dir + "%s/%s.jpg" % (mode, i)
            label_file = open(data_dir + "%s/gt_%s.txt" % (mode, i))
            label_file = label_file.readlines()

            _quad = []
            _classes = []

            for label in label_file:
                _x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3, txt = label.split(",")[:9]

                if "###" in txt:
                    continue

                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                _y0, _x1, _y1,_x2, _y2, _x3, _y3 = [int(p) for p in [_y0, _x1, _y1,_x2, _y2, _x3, _y3]]

                _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                _classes.append(1)

            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))

    def get_MLT(self):
        data_dir = os.path.join(self.root, 'MLT/')

        dataset_list = os.listdir(data_dir + "train")
        dataset_list = [l[:-4] for l in dataset_list if "jpg" in l]

        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on MLT : ", dataset_size)

        for i in dataset_list:
            img_file = data_dir + "%s/%s.jpg" % (mode, i)
            label_file = open(data_dir + "%s/gt_%s.txt" % (mode, i))
            label_file = label_file.readlines()

            _quad = []
            _classes = []

            for label in label_file:
                _x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3, lang, txt = label.split(",")[:10]

                if "###" in txt:
                    continue

                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                _y0, _x1, _y1,_x2, _y2, _x3, _y3 = [int(p) for p in [_y0, _x1, _y1,_x2, _y2, _x3, _y3]]

                _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                _classes.append(1)

            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
            
    def get_ICDAR2013(self):
        data_dir = os.path.join(self.root, 'ICDAR2013_FOCUSED/')

        dataset_list = os.listdir(data_dir + "train")
        dataset_list = [l[:-4] for l in dataset_list if "jpg" in l]

        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on ICDAR2013 : ", dataset_size)

        for i in dataset_list:
            img_file = data_dir + "%s/%s.jpg" % (mode, i)
            label_file = open(data_dir + "%s/gt_%s.txt" % (mode, i))
            label_file = label_file.readlines()

            _quad = []
            _classes = []

            for label in label_file:
                _xmin, _ymin, _xmax, _ymax = label.split(" ")[:4]

                _x0 = _xmin
                _y0 = _ymin
                _x1 = _xmax
                _y1 = _ymin
                _x2 = _xmax
                _y2 = _ymax
                _x3 = _xmin
                _y3 = _ymax

                _x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3 = [int(p) for p in [_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3]]

                _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                _classes.append(1)

            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
            
def test():
    import torchvision

    from augmentations import Augmentation_traininig
    
    dataset = ListDataset(root='/root/DB/',
                          dataset='ICDAR2015', train=True, transform=Augmentation_traininig, input_size=600, multi_scale=True)

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    count=0
    for n, (img, boxes, labels) in enumerate(dataloader):
        print(img.size(), boxes.size())
        exit()
        img = img.data.numpy()
        img = img.transpose((1, 2, 0)) * 255

        img = np.array(img, dtype=np.uint8)

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        boxes = boxes.data.numpy()
        boxes = boxes.reshape(-1, 4, 2)

        for box in boxes:
            draw.polygon(np.expand_dims(box,0), outline=(0,255,0))
        img.save('/home/beom/samba/%d.jpg' % n)

        if n==19:
            break
        
def test2():
    import torchvision

    from augmentations import Augmentation_traininig
    
    dataset = ListDataset(root='/root/DB/',
                          dataset='ICDAR2015', train=True, transform=Augmentation_traininig, input_size=600, multi_scale=True)

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    for i in range(10):
        data = dataset.__getitem__(i)
        
        random_choice = random.randint(0, len(dataset.MULTI_SCALES)-1)
        size = dataset.MULTI_SCALES[random_choice]
    
        img, boxes, labels = dataset.transform(size=size)(data['image'], data['boxes'], data['labels'])

        img = img.data.numpy()
        img = img.transpose((1, 2, 0))
        img *= (0.229,0.224,0.225)
        img += (0.485,0.456,0.406)
        img *= 255.
    
        img = np.array(img, dtype=np.uint8)

        boxes = boxes.data.numpy()
        boxes = boxes.reshape(-1, 4, 2).astype(np.int32)

        img = cv2.polylines(img, boxes, True, (255,0,0), 4)
        #cv2.imwrite('/home/beom/samba/%d.jpg' % i, img)
        
        img = Image.fromarray(img)
        img.save('/home/beom/samba/%d.jpg' % i)
    
    
#test()
