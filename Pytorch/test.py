import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw


print('Loading model..')
net = RetinaNet()

#net = torch.nn.DataParallel(net)
#cudnn.benchmark = False

#net.load_state_dict(torch.load('./checkpoint/params.pth'))
checkpoint = torch.load('weights/v1_step1/ckpt_3000.pth')

net.load_state_dict(checkpoint['net'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

img_dir = "/root/DB/ICDAR2013_FOCUSED/test/"
val_list = [im for im in os.listdir(img_dir) if "jpg" in im]

for n, _img in enumerate(val_list):
    print('Loading image...', _img)
    img = Image.open(img_dir + _img)
    w = h = 600
    img = img.resize((w,h))

    print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    x = Variable(x)
    #x = x.cuda()
    loc_preds, cls_preds = net(x)

    print('Decoding..')
    encoder = DataEncoder()
    boxes, labels, scores = encoder.decode(loc_preds.data.squeeze(0), cls_preds.data.squeeze(0), (w,h))
    
    draw = ImageDraw.Draw(img)
    
    boxes = boxes.data.numpy()
    boxes = boxes.reshape(-1, 4, 2)

    for box in boxes:
        draw.polygon(np.expand_dims(box,0), outline=(0,255,0))
    
    img.save("/home/beom/samba/pytorch_result/" + _img)

    if n==10:
        break