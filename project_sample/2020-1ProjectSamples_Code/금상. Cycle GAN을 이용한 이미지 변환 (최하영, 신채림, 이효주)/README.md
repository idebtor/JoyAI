# CycleGan을 이용한 이미지 변환



## Motivation

- 이미지의 style transfer는 pre-trained된 CNN의 feature map을 이용한다고 알고 있었는데, Cycle-consistent Adversarial Network를 이용하여 style transfer뿐만 아니라 하나의 이미지를 다른 도메인의 이미지로 변환 가능하다는 점이 매우 신기했고, 성능을 개선하기 위해 일반적인 GAN이 아닌 Cycle GAN이라는 새로운 개념을 도입했다는 점이 흥미로워 프로젝트 주제로 선정하게 되었습니다.

- 인공지능을 이용하여 미래의 데이터를 예측하거나 데이터를 분류하는 예제를 넘어 이미지 시각화를 통해 직관적인 결과 확인과 성능 비교가 가능한 주제라면 딥러닝을 처음 시작하는 사람들도 쉽게 흥미를 가질 수 있을 것이라 여겼습니다.
<p align="center"><img src="https://user-images.githubusercontent.com/47182864/82872135-93508200-9f6d-11ea-90d0-3c51b668c9a8.png" height="350px" width="650px"></p>




## Project summary

- Cycle GAN 논문을 읽고 전반적인 concept과 기존 GAN과 Cycle GAN의 차이점과 학습 algorithm을 이해합니다.

- Cycle GAN을 Pytorch로 구현한 github 오픈소스를 이용하여 facades, cityscapes, horse2zebra등과 같은 dataset을 이용하여 학습 모델을 training 시킵니다.

- Testing을 통해, 학습된 모델이 이미지를 어떻게 생성하고 변형하였는지 확인합니다



## Goals

- Cycle GAN이 무엇이며 GAN과 어떤 차이가 있으며, 성능 개선을 위해 새롭게 적용된 알고리즘과 모델이 어떤 과정을 통해 학습하는지 이해할 수 있습니다. 

- Cycle GAN을 이용하면 Input Image에 대해 흑백 사진을 컬러 사진으로, 간단한 일러스트를 구체적인 사진으로 만들어내는 것이 가능하며, 다양한 style transfer를  적용할 수 있습니다.

- 두 도메인 X, Y의 이미지가 주어졌을 때, X에 속하는 이미지를 Y에 속하는 이미지로 바꿔줄 수 있습니다.

- 모델이 원본 Input image를 특정 아티스트 스타일의 그림으로 변환하는 방법을 학습하기도 하며, 아티스트의 원본 그림을 실제 사진으로도 바꿀 수 있습니다.

## Dataset
- 이용가능한 데이터셋: [apple2orange, orange2apple, summer2winter_yosemite, winter2summer_yosemite, horse2zebra, zebra2horse, monet2photo, style_monet, style_cezanne, style_ukiyoe, style_vangogh, sat2map, map2sat, cityscapes_photo2label, cityscapes_label2photo, facades_photo2label, facades_label2photo, iphone2dslr_flower]
- 이 [링크](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)에서 zip파일로 다운 받거나, 코드 내에서 쉘을 이용하여 다운받을 수 있습니다. 

## Pre-requisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU
- CUDA CuDNN
- PyTorch
- visdom

## Getting start
### Install
1. Jupyter Notebook 또는 colab에 접속
2. git clone 명령어를 이용하여 cycleGAN 깃허브 저장소를 복제
```python
!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
```
3. os import
```python
import os
os.chdir('pytorch-CycleGAN-and-pix2pix/')
```
4. 필요한 라이브러리 설치
```python
!pip install -r requirements.txt
```
### Pretrained model 
```python
!bash ./scripts/download_cyclegan_model.sh horse2zebra
```

### Training
```python
!python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan
```

### Testing
```python
!python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```

### Visualize
```python
import matplotlib.pyplot as plt

img = plt.imread('./results/horse2zebra_pretrained/test_latest/images/n02381460_1010_fake.png')
plt.imshow(img)
```
```python
img = plt.imread('./results/horse2zebra_pretrained/test_latest/images/n02381460_1010_real.png')
plt.imshow(img)
```

---

## Youtube Link
https://youtu.be/FnAtxYOKw3E

## Presentation data
https://github.com/chy0428/CycleGAN/blob/master/final_report_cycleGAN.pdf
