# RetinaNet + TextBoxes++

RetinaNet + TextBoxes++ (pytorch >= 0.4.1)

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
- ![image](https://media.oss.navercorp.com/user/10344/files/41cde0aa-f71a-11e8-9e6d-16941677b4af)

## RetinaNet 특징
- one-stage object detection 중에서 간단하며 성능이 좋은 모델
- FPN (Feature Pyramid Network)를 사용하여 다양한 level의 feature들을 활용가능하다.
- output : 1-d score + 4d-anchor box offset
- cls loss = focal loss, loc loss = smooth L1 loss
- ![image](https://media.oss.navercorp.com/user/10344/files/54f46498-f714-11e8-9897-e0e2cf0434f7)
- ImageNet pre-trained weight initialize는 필수! -> loss가 폭발, 그냥 학습이 안된다..!
- batch norm freeze도 필수! -> freeze 시키지않으면 성능이 반 이상 하락한다. 또한 box offset은 어느정도 찾긴 하지만, classification이 엉망이다.
- freeze BN 또는 Group Norm을 사용하면 잘동작한다. 하지만 GN는 학습 속도 (group=32)가 느리며, freeze BN과 큰 성능차이가 없었다.

## Encode
1. 각 grid마다 anchor box들을 정의한다.
2. GT box와 anchor box간의 IoU를 구한다.
3. 각 anchor box는 IoU가 가장 큰 GT box로 할당된다.
4. 이 때, IoU > 0.5 : Text(label=1)  /  0.4 < IoU < 0.5 : Ignore(label=-1)  /  IoU < 0.4 : non-text(label=0) 으로 할당된다.
5. 아래 식을 이용해 anchor box coordinate encode를 수행한다.
![image](https://media.oss.navercorp.com/user/10344/files/7f1816b4-f7e8-11e8-94e8-7aad5cbe5330)

---

## Experiments1 : loc output channel
- loc output의 경우 (4+8)-d이다.
- ICDAR2015에서 rect box를 위한 앞의 4개의 offset은 사용하지 않고, 뒤의 8개의 quad offset만 사용한다.
- loc output을 12-d vs 8-d 로 실험을 해본 결과 거의 비슷한 성능을 얻었다.

## Experiments2 : backbone
- 우선 이전 실험을 통해 resnet50보다 se-resnet50이 더 좋은 성능을 얻었었다.
- se block을 default로 사용을 하되 se-renset50 vs se-resnet101을 실험해보았다.

| Backbone     |  loc output | FPN range | IC15 Hmean (%) | 
|:--------:  | :-----:   | :----:      |  :-----:  |
| SE-Renset50    | 8-d  |  C3~C5  | 0.7710  |
| SE-Renset101   | 8-d  |  C3~C5  | 0.7912  |

## Experiments3 : FPN range
- RetinaNet에서는 FPN에 사용되는 부분으로 C3~C5를 사용한다.
- ![image](https://media.oss.navercorp.com/user/10344/files/a811bbe2-f71b-11e8-9c62-ef3e1cd8617c)
- 하지만 Text의 경우 크기가 작은 경우가 많기 때문에, low-level의 feature를 활용하도록 C2~C5로 수정하여 실험을 진행하였다.

| Backbone     |  loc output | FPN range | IC15 Hmean (%) | 
|:--------:  | :-----:   | :----:      |  :-----:  |
| SE-Renset50    | 8-d  |  C3~C5  | 0.7710  |
| SE-Renset50   | 8-d  |  C2~C5  | 0.7916  |
- 그 결과 2% 정도의 성능 향상이 있었다.

## Experiments4 : Multi-scale Training
- Multi-scale Training은 YOLOv2에 사용된 학습 기법으로, 다양한 scale에 강인해지도록하는 효과가 있다.
- ex) 1-iter : [batch size, 608, 608, 3] -> 2-iter : [batch size, 960, 960, 3] -> 3-iter : [batch size, 768, 768, 3]

| inference input size     |  single scale | multi scale |
|:--------:  | :-----:   | :----:      |
| 768x768    | 0.7474  |  0.7817  |
| 960x960   | 0.7822  |  0.7856  |
| 1280x1280   | 0.7633  |  0.7977  |
- 적용 결과 다양한 scale에 대해 강인해졌지만, 기대했던것 만큼 엄청난 성능 향상은 없었다.

## Experiments5 : new Anchor Boxes
- FPN에서의 low level feature에서는 큰 글자를, high level feature에서는 작은 글자를 찾도록 설계 되어있다.
- 기존 anchor box의 경우 C2:16x16 / C3:32x32 / C4:64x64 / C5:128x128 / C6:256x256 / C7:512x512 로 정의하며, aspect ratio는 각각 [1, 2, 3, 5, 0.5, 0.33, 0.2] 로 사용하였다.
- anchor box도 task에 따라 다른 특성을 보일 것이므로, K-means clustering으로 ICDAR2013, 2015 dataset에 대해 box들의 scale과 aspect ratio를 뽑아보았다. (k=6)

| version  | C2 | C3 | C4 | C5 | C6 | C7 | aspect ratio |
|:----:  | :----:   | :----: | :----: | :------: | :-----: | :------: | :------: |
| original | 16x16  |  32x32 |  64x64 |  128x128 |  256x256|  512x512 | 1, 2, 3, 5, 1/2, 1/3, 1/5 |
| new | 30x30  |  70x70 |  120x120 |  250x250 |  320x320|  450x450 | 1, 2, 3, 5, 1/2, 1/5 |

- 이렇게 K-means clustering으로 ICDAR dataset에 맞는 anchor box들을 뽑아서 새롭게 학습을 하였다.

| version     |  IC15 Hmean (%) |
|:--------:  | :-----:   |
| original (16, 32, 64, 128, 512)    | 0.7977  |
| new (30, 70, 120, 250, 320, 450)   | 0.7752  |

- ICDAR에 최적화된 anchor box를 뽑아 더 개선되는줄 알았으나, 오히려 2%의 성능 하락이 있었다.
- regression으로 anchor box에서 얼마나 늘리고 줄일지에 대한 offset을 예측하는 것이기 때문에 최대한 anchor scale의 범위가 넓은 것이 유리하다고 판단된다. (new anchor에서는 anchor scale의 min/max 범위가 줄어들었기 때문에)

## Experiments6 : Focal_loss vs OHEM+CE_loss
- RetinaNet을 기반으로 하였기 때문에 default로 focal loss를 사용하였다.
- 하지만 focal loss가 RetinaNet 구조에서만 잘동작하는(?) 경향이 있다고 한다. (다른 여러 Ojbect Detection 논문에서 focal loss로 실험을 해봤는데 별차이가 없다고들 많이 언급되어있다.)
- 그래서 classification loss에 대해 성능 비교를 해볼 필요가 있었다.
- focal loss : Binary Cross Entropy 기반
- OHEM + SoftMax Cross Entropy loss (2-class -> text or non-text)
- SSD_pytorch 코드에서 loss부분을 가져와서 사용했지만, classification score가 모든 anchor에 대해 같은 값을 갖는 결과가 나왔다...?!
- 2-classes이지만 gt label에는 text=1만 존재하기 때문에 classification에 overfitting이 발생한 것이 아닐까 생각한다.

---

## 최종 결과
- [실험 정리/성능 표](https://docs.google.com/spreadsheets/d/1Iq5JlMM95Xp5f6kLSZPLImi_9QcS9usRuMH5vlKIZU8/edit?usp=sharing)

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
  - ![image](https://media.oss.navercorp.com/user/10344/files/bfc5140e-f898-11e8-8682-8cadbde2efab)
  - ![image](https://media.oss.navercorp.com/user/10344/files/fcfb8146-f898-11e8-8f0c-5b05c067e4fd)
  - ICDAR2015
  - ![image](https://media.oss.navercorp.com/user/10344/files/57c19340-f7ef-11e8-8998-c76713d92d66)
  - ![image](https://media.oss.navercorp.com/user/10344/files/7a2f4c60-f7ef-11e8-8de3-865559e57fe2)

## Bad case
- vertical box에 대해 약하다..!
- GT box와 비교했을 때, text 영역을 잘감싸는 fitting이 부족하다.
- 길이가 긴 Text에 대한 처리 능력이 부족하다.
- 애매한 또는 처음 보는 text에 대한 대처 능력이 부족하다.
- Red : Prediction  /  Green : GT  /  Yellow : Don't Care
  - ICDAR2013
  - ![image](https://media.oss.navercorp.com/user/10344/files/2b54d362-f899-11e8-929e-6697ce03af23)
  - ![image](https://media.oss.navercorp.com/user/10344/files/517ef7a2-f899-11e8-87e2-62c419474152)

  - ICDAR2015
  - ![image](https://media.oss.navercorp.com/user/10344/files/a9b98afe-f7ef-11e8-8132-5e2608ff7e11)
  - ![image](https://media.oss.navercorp.com/user/10344/files/d936523a-f7ef-11e8-96a1-596614f4badf)

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
