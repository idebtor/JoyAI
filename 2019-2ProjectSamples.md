# 19-2 우수 프로젝트

## Unsupervised learning을 이용한 영상 개선(김혜원, 권은혁)


### 프로젝트를 하게 된 계기
- 영상개선 문제는 clean 한 target영상이 필요한 supervised learning을 통해서만 될 수 있는것이라고 생각했었는데, unsupervised learning이라는 새로운 접근으로 좋은 성능을 보이는 이 프로젝트가 신기했으며 학습데이터가 많이 필요하지 않는 unsupervised learning에 대해 더 알아보고 싶어서 결정하게 되었습니다.


### 프로젝트 개요
- 이 프로젝트는  "Deep image prior" 논문과 관련 구현 github 코드를 활용하여 구현할 예정입니다.
- 기존의 'inpainting' 코드를 응용해 사용자가 영상에서 지우고싶은 글씨를 지울 수 있도록 새롭게 customize 해보았습니다.


### 프로젝트 과정
1. 관련 library import
2. image pat설정 및 upload
3. custom mask 제작 및 저장
4. optimizer, iteraion, loss function 등 parameter 설정
5. optimize
6. 결과 비교 및 저장


### 프로젝트 기대효과
- random한 noise 영상에 대해 학습없이 clean한 결과영상을 얻을 수 있음.
- super resolution, denoising, inpainting과 같은 여러 응용에 이용될 수 있음.
- 사용자가 직접 마스크를 제작하여 영상에서 마스크 부분을 inpainting시킬 수 있음 ⭐️


### 영상 링크
- [YouTube](https://www.youtube.com/watch?v=q5ZlfIDGW-Y&feature=youtu.be)


### 프로젝트 파일
- [inpainting_with_custom_mask.ipynb -프로젝트코드](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fjzt8zq92s8a6wf%2Fk3o8pilkg6ws%2Finpainting_with_custom_mask.ipynb)
- [DIP.ipynb -환경설정 및 프로젝트 소개](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fjzt8zq92s8a6wf%2Fk3o8q3333qwx%2FDIP.ipynb)
- [deepimageprior20191202T092314Z001.zip -전체코드 zip](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fjzt8zq92s8a6wf%2Fk3o8r7w0hi3p%2Fdeepimageprior20191202T092314Z001.zip)
-----------------------------------


## Neat algorithm을  사용해서 Flappy Bird 게임하는 방법 배우기(김혜영, 성은채, 박민우)

### 프로젝트를 하게된 계기
- 인공지능에게 구체적이지 않고 매우 간단한 명령과 모델을 제공하였을 때, 스스로 훈련하고 학습해서 놀라운 성과를 보여주는 부분에 흥미를 느껴 시작하게 되었습니다.


### 프로젝트소개 
- Neuro Evolution of  Augmenting Topologies(NEAT) 신경망 모델을 구성하는 프로그래밍을 이해하고, 이 신경망 모델을 통해 강화 학습시켜 컴퓨터가 스스로  flappy bird 게임을 플레이 하는 방법을 배우게 하는 프로젝트 입니다.


### 프로젝트 개요
- 이 게임을 학습하기 위해 필요한 Neat algorithm을 활용한 오픈소스를 github에서 다운 받습니다 
- Neat algorithm 을 사용해 컴퓨터를 계속 training하게 하여 컴퓨터가 Flappy Bird 게임하는 방법을 학습합니다


### 기대효과
- 이 프로젝트를 통해서, 컴퓨터가 게임하는 방법을 어떻게 스스로 학습하는지 알 수 있습니다.
- 컴퓨터에게 학습을 오래 시킬수록 더 좋은 학습모델을 생성시켜 컴퓨터 스스로가 게임을 더 잘 진행하게 됩니다.


### 영상 링크
- [영상1: 문제 제기, AI 소개](https://youtu.be/Zry28z3g4MY)
- [영상2: 코드 설명, 결과 분석](https://youtu.be/2wrC_f_wWNk)

### 프로젝트 파일
1. 개발 환경 구성 및 실행 방법
    - [README.rtf](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fj6zrjxq3baz7ol%2Fk3p9a9bu1a54%2FREADME.rtf)
    - [README.png ](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fj6zrjxq3baz7ol%2Fk3p913n52qzl%2FREADME.png)

2. 실행파일 
    - [Flappy.zip](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fj6zrjxq3baz7ol%2Fk3p8k8vjfdis%2FFlappy.zip)


3. 주피터노트북 파일
    - [Flappy_Final.ipynb](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fjzwu90twvot5ak%2Fk3p91wgwjido%2FFlappy_Final.ipynb)


-----------------------------------
##  랜덤 포레스트 기법을 통한 비행기 출발 지연 예측(안병웅, 최은평)


### 프로젝트를 하게 된 계기
- 실제로 AI를 통해 미래의 대한 예측이 가능한지에 대해 항상 궁금했었고, 확인하고 싶었는데, 이번 프로젝트를 통해 많은 데이터를 통해 미래의 일을 예측할 수 있다는 것을 직접 해보고 싶어 비행기 출발 지연 예측 이라는 주제를 선택하였다.


### 프로젝트 개요
- 인공지능을 학습시키기에 필요한 데이터를 선별한다.
- 데이터 모델을 나누어 하나는 인공지능을 학습시키고, 하나는 인공지능이 제대로 예측하는지 평가하는 모델로 나눈다.
비행기 출항은 당연히 노이즈 데이터가 존재할 수 있기 때문에 Random Forest 기법을 사용하여 인공지능을 학습시킨다.
 

### 프로젝트 기대효과
- 평소 많은 사람들이 비행기 지연으로 시간을 낭비한다. 따라서 인공지능을 이용하여 미리 예측해 이러한 불편함을 줄일 수 있다.
 

### 영상 링크
- [영상링크](https://www.youtube.com/watch?v=n6yA1z3PIj0&feature=youtu.be) 
 

### 프로젝트 파일
- [10조_프로젝트_압축파일.zip - 프로젝트 주피터 노트북,데이터, 개발환경](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fjsjvuwygwzf2p3%2Fk3p2upmnll2g%2F10%E1%84%8C%E1%85%A9_%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3_%E1%84%8B%E1%85%A1%E1%86%B8%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF.zip)


-----------------------------------
## 수강과목 분석을 통한 전공 예측 프로그램(윤희원, 김주찬, 김유빈)

- 한동대학교 1학년 학생들의 1, 2학기 수강과목을 통해 어떠한 1, 2 전공을 선택할지 예측하는 프로그램입니다. 

### 프로젝트를 하게 된 계기
- 한동대학교는 1학년 기간동안 다양한 수업들을 듣고, 2학년이 될 때 자신의 전공을 선택합니다. 학교 측 입장에서는 기존의 1학년 2학기 말 학생들이 직접 작성한 희망전공을 통해 학생들의 2학년 전공을 파악하고, 그에 따라 예산을 준비합니다. 이 프로젝트를 통하여 1학년 학생들의 2학기 수강신청 기간 직후, 즉 기존보다 빠른 시점에서 예산 편성, 다음 학년 준비 등의 업무를 수행할 수 있을 것이라 생각합니다. 그 밖에도 학생들로 하여금 스스로 자신이 관심있어 하는 과목과 실제 학교 내의 전공들 간의 관계를 파악하여 자신의 전공을 선택하는데 도움을 줄 수 있을 것이라 생각합니다.

 
### 프로젝트 개요
    1. '모두를 위한 인공지능의 활용’ 수업 github에서 JoyML11-3BatchGD.ipynb 파일을 다운로드 한다.
    2. MNIST BGD 모델 코드를 활용하여 우리가 실제 학습하고자하는 신경망의 형태로 변환한다.
    3. 18학번들의 수강과목 데이터를 다운받아서 필요한 정보들만 추출하여 CSV 형태로 가공한다.
    4. 가공된 데이터를 Jupyter Notebook으로 읽어들인 후 학습시킨다.
    5. 테스트용 데이터를 활용하여 예측 정확도를 확인한다.


### 기대효과
- 전공예측 프로그램을 통해 학교측은 예산 편성을 미리할 수 있어 발빠르게 움직일 수 있고, 학생측은 자신이 들은 수강과목을 통해 전공을 판단하여 한 단계 미리 나갈 수 있습니다.


### 영상 링크
- [영상 링크](https://www.youtube.com/watch?v=Ohz3Ijl4Jhw)


### 프로젝트 파일
- [프로젝트 설명 파일 : 설명_파일.zip](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fjzwyxnz44qe2q3%2Fk3or4aprcdk6%2F%EC%84%A4%EB%AA%85_%ED%8C%8C%EC%9D%BC.zip)
- [프로젝트 코드 (School prediction1.ipynb, joy.py, BatchGD.ipynb 포함) : 코드.zip](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fjzwyxnz44qe2q3%2Fk3or5ifsp4f2%2F%EC%BD%94%EB%93%9C.zip)
- [프로젝트에 맞게 가공한 학생 데이터 (수강과목, 학부)](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fjzwyxnz44qe2q3%2Fk3or67y8hreh%2F%EC%88%98%EA%B0%95%EA%B3%BC%EB%AA%A9_%ED%95%99%EB%B6%80_CSV.zip)


-----------------------------------

## 유전자 정보를 이용한 난소암 항암제 내성 예측해보기 (송하림, 이지현)
- TensorFlow를 이용한 학습 모델을 활용하여서 난소암의 백금 성분 항암제 내성을 예측하는 프로젝트입니다.
 

### 프로젝트를 하게 된 계기
- 저희 조는 생명 전공생 한 명, 전산 전공생 한 명으로 이루어져 있습니다. 인공지능 모델을 이용하여 난소암 항암제 내성 여부를 예측하는 것은 각자의 전공과 관심 분야를 모두 살릴 수 있는 주제라고 생각되어, 본 프로젝트를 선정하게 되었습니다.
- 또한, CNN이나 RNN 등의 복잡한 인공지능 모델을 사용하는 것이 아닌, 수업 시간에 배웠었던 Fully Connected Layer 만을 이용한 간단한 모델을 만들어 문제를 해결할 예정입니다. 이 과정을 통해 수업 때 배웠던 간단한 모델 만으로도 자신이 평소에 흥미를 가지고 있었던 전문 분야의 문제를 해결할 수 있다는 것을 공유하고 싶었습니다.
 

### 프로젝트 개요
 - TCGA 에서 유전자 정보에 대한 공공 데이터를 다운받고, 전처리 과정을 거쳐 Train / Validation / Test Dataset을 생성합니다.
- 로딩한 Train Dataset을 이용하여 간단한 ANN 모델을 학습시킵니다.
- 다양한 메트릭을 이용하여 학습된 모델의 예측 정확도를 평가합니다.


### 프로젝트 기대효과
- 수업 때 배웠던 간단한 ANN 만으로도 자신이 평소에 흥미를 가지고 있었던 전문 분야의 문제를 해결할 수 있음을 알 수 있습니다.
- 수업 때 배웠던 내용을 전산 외의 다른 분야의 연구에 적용할 수 있음을 체험합니다.
- 항암제 (백금 성분) 저항성을 가지게 되는 환자들을 미리 예측할 수 있다면, 그들에게 다른 치료법을 시행할 수 있어 환자들의 시간 및 비용이 절감됩니다.
 

### GitHub Link (Souce Code + Pre-Processed Dataset + 배경 설명 PPT)
- https://github.com/jihyunlee96/JoyAI_Team16_Cancer_Drug_Resistance_Prediction

 
 ### 영상 링크
- [배경 설명 / 환경 설정 / 데이터셋 설명](https://youtu.be/UNBisQu6jw0)
- [코드 설명 (1)](https://youtu.be/CAW1bLGndew)
- [코드 설명 (2)](https://youtu.be/oUKI4E7Q2L4)


### 프로젝트 파일
- [GitHub Link](https://github.com/jihyunlee96/JoyAI_Team16_Cancer_Drug_Resistance_Prediction)
- [GitHub Repository 압축 파일](http://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fjks90klk7p477h%2Fk3omxgorqm24%2FJoyAI_Team16_Cancer_Drug_Resistance_Prediction.zip)


-----------------------------------

## 스타(연예인) 사진으로 다양한 합성사진 만들기(박준혁, 조경민, 박범준)

 
### 프로젝트를 하게 된 계기
- 조원들 모두 영상처리쪽에 관심이 있었는데, 이 모델은 한가지의 응용이 아닌, 성별 바꾸기, 표정변화 등 다양한 응용을 한번에 할 수 있었기에 이 프로젝트를 설정하게 되었습니다.

 
### 프로젝트 개요
![test](https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjyselhvwmic3cs%2Fje3s9tpymbg7m8%2Fk3egfk8hkigp%2FScreen_Shot_20191125_at_10.15.25_PM.png)

- 이 프로젝트는 GAN 모델을 바탕으로 "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation" 논문과 관련 구현 github 코드를 활용하여 구현할 예정입니다.

1) 논문 저자들이 제공한 pretrained model 파일을 활용하여 다양한 응용의 weight 등의 정보를 사용한다.

2) 각 응용에 대한 정보를 활용하기 위하여 input 영상의 size 조절, 인물 위치 등 제약조건을 만족시켜 준다.

2) User에게서 입력받은 영상을 기준으로 다양한 응용에 대한 결과 영상을 생성한다.

 
### 기대효과
1. 한가지의 학습 모델이 여러가지 응용에 활용될 수 있다는 것을 보임.
2. 다양한 응용중 하나인 표정 변화 합성을 활용하여 주어진 영상에서 사용자가 원하는 표정으로 바꿀 수 있다.

프로젝트 Github 페이지 (프로젝트 파일, 동영상링크, 사용법, jupyter notebook파일 전부 포함)


### 프로젝트 파일
- https://github.com/pjh1023/TeamProject-Team3-StarGAN



### 영상 링크
- [1부](https://youtu.be/qndhOzTewk0) 
- [2부](https://youtu.be/W8OTKgyiabc) 
 

------------------------

## Genetic algorithm을 이용한 뱀게임을 하는 인공지능 프로그램입니다. (박성환, 장현우)

### 프로젝트를 하게 된 계기
- 파이썬에 pygame이라는 게임 개발 도구가 있다는 것을 보고, 뱀게임을 만들어 보았습니다. 뱀게임을 사람이 하는 것이 아니라, 인공지능이 스스로 학습하고 훈련하면 어떤 결과가 나올지 궁금해서 프로젝트를 시작하게 되었습니다.

 

### 프로젝트 개요
1. pygame인 snake 구현
2. 인공지능에 방향과 센서 구현
3. 먹이를 향해 가까워질 때 +1점, 먹이로부터 멀어질 때 -1.5점, 먹이를 먹었을 때 10점 부여
4. 진화 알고리즘(crossover, mutation)을 바탕으로 인공지능을 강화학습 시킨다.

 

### 기대효과
1. 뱀게임에 대한 강화 학습을 오래 시킬수록 더 높은 점수를 획득하는 모델이 나온다.
2. 인공지능이 발전하는 과정을 볼 수 있다.

 
### 프로젝트 파일
 - [프로젝트 Github 페이지 (프로젝트 파일, 동영상링크, 사용법, jupyter notebook파일 전부 포함)](https://github.com/hw78dh/hello)
 

### 영상 링크
- [YouTube](https://www.youtube.com/watch?v=EgYLjnemlok)
