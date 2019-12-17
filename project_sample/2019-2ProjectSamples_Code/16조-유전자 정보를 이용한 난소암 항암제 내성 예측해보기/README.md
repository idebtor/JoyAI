# 유전자 정보를 이용한 난소암 항암제 내성 예측
TensorFlow와 Keras 기반의 DNN 모델을 사용하여 난소암의 백금 성분 항암제 내성을 예측하는 프로젝트입니다.

## Contributors
Biodata Lab @ HGU, Harim Song, Jihyun Lee

![Gene](https://user-images.githubusercontent.com/37920573/69950538-d12c9a80-1536-11ea-9b27-2b0395c82090.jpeg)

## Motivation
- 저희 조는 생명 전공생 한 명, 전산 전공생 한 명으로 이루어져 있습니다. 인공지능 모델을 이용하여 난소암 항암제 내성 여부를 예측하는 것은 각자의 전공과 관심 분야를 모두 살릴 수 있는 주제라고 생각되어, 본 프로젝트를 선정하게 되었습니다.
- 또한, CNN이나 RNN 등의 복잡한 인공지능 모델을 사용하는 것이 아닌, 수업 시간에 배웠었던 Fully Connected Layer 만을 이용한 간단한 모델을 만들어 문제를 해결할 예정입니다. 이 과정을 통해 수업 때 배웠던 간단한 모델 만으로도 자신이 평소에 흥미를 가지고 있었던 전문 분야의 문제를 해결할 수 있다는 것을 공유하고 싶었습니다.


## Project Summary
1.  TCGA 에서 유전자 정보에 대한 공공 데이터를 다운받고, 전처리 과정을 거쳐 Train / Validation / Test Dataset을 생성합니다.
2.  Train Dataset을 이용하여 간단한 DNN 모델을 학습시킵니다.
3.  다양한 메트릭(Accuracy, Specificity, Sensitivity)을 이용하여 학습된 모델의 예측 정확도를 평가합니다.


## Project Goals
- 수업 때 배웠던 간단한 DNN 만으로도 자신이 평소에 흥미를 가지고 있었던 전문 분야의 문제를 해결할 수 있음을 알 수 있습니다.
- 수업 때 배웠던 내용을 전산 외의 다른 분야의 연구에 적용할 수 있음을 체험합니다.
- 항암제 (백금 성분) 저항성을 가지게 되는 환자들을 성공적으로 예측할 수 있다면, 그들에게 다른 치료법을 시행할 수 있어 환자들의 시간 및 비용이 절감될 것을 기대할 수 있습니다.

## Dataset Reference
The Cancer Genome Atlas (TCGA) Datasets  
http://software.broadinstitute.org/software/igv/tcga

## Pre-Requisites
- tensorflow 1.10.0
- keras
- numpy
- pandas
- sklearn

## Setup Guide
1. 수업시간에 배웠던 방법으로 Anaconda와 Jupyter Notebook을 설치합니다.  
2. Conda 가상 환경을 생성합니다. 
```
$ conda create -n <가상환경 이름>
```
3. Conda 가상 환경에 접속합니다. 
```
$ conda activate <가상환경 이름>
```
4. 필요 라이브러리들을 설치합니다. 
```
$ pip install tensorflow==1.10.0 numpy pandas keras sklearn
```
5. Jupyter Notebook을 실행시키고, `Cancer_Drug_Resistance_Prediction.ipynb` 파일을 엽니다.

## Youtube Links
1. 배경 설명 / 환경 설정 / 데이터셋 설명  
https://youtu.be/UNBisQu6jw0  
2. 코드 설명 (1)  
https://youtu.be/CAW1bLGndew  
3. 코드 설명 (2)  
https://youtu.be/oUKI4E7Q2L4  

