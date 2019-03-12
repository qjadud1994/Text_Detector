# Text Detector for OCR

RetinaNet + TextBoxes++

## How to use
- single scale training
```
#step1 : SynthText, 608x608
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset=SynthText --batch_size=16 --input_size=608 --logdir=logs/single_step1/ --save_folder=weights/single_step1/ --num_workers=6

#step2 : ICDAR2013 / 2015, 768x768
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset=ICDAR2015 --batch_size=16 --input_size=768 --logdir=logs/single_step2/ --save_folder=weights/single_step2/  --num_workers=6 --resume=weights/single_step1/ckpt_40000.pth
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset=ICDAR2013/MLT --batch_size=16 --input_size=768 --logdir=logs/single_step2/ --save_folder=weights/single_step2/  --num_workers=6 --resume=weights/single_step1/ckpt_40000.pth

#step3 : ICDAR2013 / 2015, 960x960
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset=ICDAR2015 --batch_size=8 --input_size=960 --logdir=logs/single_step3/ --save_folder=weights/single_step3/  --num_workers=6 --resume=weights/single_step2/ckpt_10000.pth
```

- multi scale training
```
#step1 : SynthText, MS(608x608 ~ 960x960)
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset=SynthText --batch_size=12 --multi_scale=True --logdir=logs/multi_step1/ --save_folder=weights/multi_step1/ --num_workers=6

#step2 : ICDAR2013 / 2015, MS(608x608 ~ 960x960)
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset=ICDAR2015 --batch_size=12 --multi_scale=True --logdir=logs/multi_step2/ --save_folder=weights/multi_step2/ --num_workers=6 --resume=weights/multi_step1/ckpt_40000.pth
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset=ICDAR2013/MLT --batch_size=12 --multi_scale=True --logdir=logs/multi_step2/ --save_folder=weights/multi_step2/ --num_workers=6 --resume=weights/multi_step1/ckpt_40000.pth

#step3 : ICDAR2013 / 2015, MS(960x960 ~ 1280x1280)
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset=ICDAR2015 --batch_size=6 --multi_scale=True --logdir=logs/multi_step3/ --save_folder=weights/multi_step3/ --num_workers=6 --resume=weights/multi_step2/ckpt_10000.pth
```

- For Evaluation
```
CUDA_VISIBLE_DEVICES=0 python eval.py  --input_size=1280 --nms_thresh=0.1 --cls_thresh=0.4 --dataset=ICDAR2015 --tune_from=weights/multi_step3/ckpt_4000.pth --output_zip=multi_step3_zip
```

---

## TextBoxes++ 특징
- TextBoxes와 구조는 같지만, Quad Box를 위한 offset이 추가되었다.
- 4d-anchor box(xywh) offset -> (4+8)-d anchor box(xywh + x0y0x1y1x2y2x3y3) offset
- last conv : 3x5 -> quad box에 최적화된 receptive field를 갖게 하기 위함

## RetinaNet 특징
- one-stage object detection 중에서 간단하며 성능이 좋은 모델
- FPN (Feature Pyramid Network)를 사용하여 다양한 level의 feature들을 활용가능하다.
- output : 1-d score + 4d-anchor box offset
- cls loss = focal loss, loc loss = smooth L1 loss
- ImageNet pre-trained weight initialize는 필수! -> loss가 폭발, 그냥 학습이 안된다..!
- batch norm freeze도 필수! -> freeze 시키지않으면 성능이 반 이상 하락한다. 또한 box offset은 어느정도 찾긴 하지만, classification이 엉망이다.
- freeze BN 또는 Group Norm을 사용하면 잘동작한다. 하지만 GN는 학습 속도 (group=32)가 느리며, freeze BN과 큰 성능차이가 없었다.

## Encode
1. 각 grid마다 anchor box들을 정의한다.
2. GT box와 anchor box간의 IoU를 구한다.
3. 각 anchor box는 IoU가 가장 큰 GT box로 할당된다.
4. 이 때, IoU > 0.5 : Text(label=1)  /  0.4 < IoU < 0.5 : Ignore(label=-1)  /  IoU < 0.4 : non-text(label=0) 으로 할당된다.
5. 아래 식을 이용해 anchor box coordinate encode를 수행한다.

---

## 최종 결과

| Framework   | Dataset     | Hmean    |   backbone  | input size | training scale | cls thresh | nms thresh  | iter    | weights |
|:--------:   | :----:   |  :--------: |  :------:  |  :----------:  |  :------:  |  :------:   |:-----:  |:-----:  | :-----:  |
| Tensorflow | ICDAR2013  |  0.8107   | se-resnet50 |   768x768    |  960x960   |    0.3     |   0.1      | Synth(62k) + IC15(17k) | [link](https://drive.google.com/open?id=1uZtCjyZ4vx9RpXaEuU2UEJIkxY8Kddfp) |
| Tensorflow | ICDAR2015  |  0.7916   | se-resnet50 |  1280x1280 |  960x960   |    0.3     |   0.1      | Synth(62k) + IC15(17k) | [link](https://drive.google.com/open?id=1uZtCjyZ4vx9RpXaEuU2UEJIkxY8Kddfp) |
| Pytorch | ICDAR2013  |  0.8298   | se-resnet50 |   multi    |  Multi scale   |    0.5     |   0.35      | Synth(35k) + IC13+MLT(7k) | [link](https://drive.google.com/open?id=1pzwDnC3C2nXtwYe0A_tkMlWwf6BJAve0) |
| Pytorch | ICDAR2015  |  0.8065   | se-resnet50 |  1280x1280 |  Multi scale   |    0.4     |   0.20      | Synth(35k) + IC15(4k) | [link](https://drive.google.com/open?id=1mDNS8RfFExjXTg-7cmU725n4P_Ma2If4) |

## Good case
- Quad box에도 잘 대응을 하며, 다양한 크기의 box들도 잘찾아낸다. 
- Red : Prediction  /  Green : GT  /  Yellow : Don't Care
  - ICDAR2013
  - ICDAR2015
  
## Bad case
- vertical box에 대해 약하다..!
- GT box와 비교했을 때, text 영역을 잘감싸는 fitting이 부족하다.
- 길이가 긴 Text에 대한 처리 능력이 부족하다.
- 애매한 또는 처음 보는 text에 대한 대처 능력이 부족하다.

---

## Experiments
- v1 : 2-classes, focal loss, loc=12-d, SGD, loss alpha=0.2, FPN [C3-C5] , anchor scale [32x32-512x512]
- v2 : 1-class, focal loss, loc=12-d, SGD, loss alpha=0.2, FPN [C3-C5] , anchor scale [32x32-512x512]
- v3 : 2-classes, focal loss, loc=8-d, SGD, loss alpha=0.5, FPN [C3-C5] , anchor scale [32x32-512x512]
- v4 : 1-class, focal loss, loc=8-d, SGD, loss alpha=0.5, FPN [C3-C5] , anchor scale [32x32-512x512]
- v5 : 1-class, focal loss, loc=8-d, SGD, loss alpha=0.5, FPN [C2-C5] , anchor scale [16x16-512x512]
- v6 : 1-class, focal loss, loc=8-d, SGD, loss alpha=0.5, FPN [C2-C5] , anchor scale [30x30-450x450]

## Todo list:
- [x] validation infernece image visualization using Tensorboard
- [x] add augmentation ( + random crop)
- [x] Trainable BatchNorm -> Not work!
- [x] Freeze BatchNorm -> work!
- [x] GroupNorm -> work!
- [x] add vertical offset
- [x] make SynthText tfrecord
- [x] make ICDAR13 tfrecord
- [x] make ICDAR15 tfrecord
- [x] add evaluation code (IC13, IC15)
- [x] use SE-resnet backbone
- [x] (Experiment) change anchor boxes scale half size -> worse result
- [x] (Experiment) FPN use C2~C5 -> little improvement
- [x] 1-class vs 2-classes -> no big difference
- [x] (binary) focal loss vs OHEM+softmax  ->  fail
- [x] QUAD version NMS (numpy version)
- [x] QUAD version Random Crop
- [x] (try) loc head : 4+8 -> 8  -> no big difference
- [x] Multi-scale training (only pytorch)
- [x] get anchor boxes using K-means clustering
- [x] change upsample function for 600 input (only pytorch)

## Detection papers
- [x] TextBoxes, TextBoxes++
- [x] SSTD
- [x] SSD, DSSD
- [x] YOLO (v1, v2, v3)
- [x] Faster RCNN + FPN, Mask RCNN
- [x] RetinaNet
- [ ] RefineDet
- [x] MegDet
- [ ] IncepText
- [ ] Feature Enhancement Network
- [ ] Learninig NMS

## Environment

- os : Ubuntu 16.04.4 LTS <br>
- GPU : Tesla P40 (24GB) <br>
- Python : 3.6.6 <br>
- Tensorflow : 1.10.0
- Pytorch : 0.4.1
- tensorboardX : 1.2
- CUDA, CUDNN : 9.0, 7.1.3 <br>
