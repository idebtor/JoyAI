# 19-1 프로젝트 

## Tensorflow와 선형회귀를 활용하여 silver to gold price 예측 프로젝트.(김정빈, 김하림, 안재욱)

### 프로젝트 개요
- 플라스크 웹 서버를 통하여 접속할 수 있으며, 은 가격 정보를 입력하면 그에 상응하는 금 가격을 계산하여 출력합니다. 사용자는 이 정보들을 읽고 둘의 상관관계와 투자에 관하여 도움을 받을 수 있습니다.

### 기대 효과
  1. 은을 통해 금의 가격 예측
  2. 금시세와 은값에 의한 예측된 금가격 정보를 비교하여 시장의 흐름 파악에 참고 자료로 활용 가능.
  3. 지난 1년 동안의 daily 데이터셋을 통해 보다 정확한 예측이 가능함.


###  특징
  1. COMEX(뉴욕상품거래소) 기준으로  달러/트로이온스(US dollar per troy ounce) 단위에 따른 정보 예측.
  2. 다중 선형 회귀가 아닌 단순 선형 회귀 활용으로 그에 따른 코드의 쓰임을 파악하는데 도움.

### 프로젝트 영상
- [Youtube](https://www.youtube.com/watch?v=Vbd7WszNr4U&feature=youtu.be)

--------------------------------------

## MCTS를 활용한 인공지능 TicTacToe 개발

장재근(21400659) / 이화평(21800608)

### 프로젝트 소개 
- 몬테카를로 트리 탐색(MCTS) 및 UCT 알고리즘을 이용한 인공지능 TICTACTOE게임 프로그램 제작


### 프로젝트 개요
- 몬테 카를로 트리 탐색에 대한 개념 이해
- 파이썬을 이용하여 Tic-Tac-Toe 게임 구현
- 파이썬을 이용하여 몬테카를로 트리 탐색 코드 구현
- 파이썬을 이용하여 UCT (Upper Confidence Bounds of Tree) 알고리즘 구현
- 게임 실행 및 확인

### 동영상 링크
- [몬테카를로 및 몬테카를로 트리 서치에 대한 이해](https://www.youtube.com/watch?v=6-cjW6X9HAI)
- [MCTS 및 UCT 알고리즘 직접적 구현 및 시연](https://www.youtube.com/watch?v=kSZ0Ovu7nug)
 

---------------------------------------
## Generative Adversarial Networks (GANs) 모델(정산, 최진아)

### 프로젝트 소개
Generative Adversarial Networks (GANs) 모델을 직접 구성, 프로그래밍해보고 모델을 사용하여 MNIST Data, Keras, Tensorflow를 이용해 숫자의 손글씨 이미지를 생성하는 모델을 만들어본다.

### 다루게 될 기술
- numpy
- MNIST Data
- Tensorflow
- GAN
- Keras
- tqdm

### 프로젝트 개요
- MNIST 데이터셋을 사용해 train, test Data set을 다운받음
- Generator Discriminators 네트워크를 구축
- 일정 수의 epoch 마다 생성된 이미지를 저장하는 함수 작성
- Training

### 프로젝트 결과물
- [영상 링크](https://youtu.be/84Ythd2f97M)


---------------------------------------

## 다음날 주가 예측하기(홍순찬)

### 프로젝트 소개
- RNN 알고리즘을 사용하여 과거 주가데이터 학습한 후, 다음날 주가 예측하기

### 다루게 될 기술
- numpy
- tensorflow
- matplotlib
- os

### 프로젝트 개요
- 과거의 주가 데이터를 학습
- 학습용/ 테스트용 데이터를 사용
- 검증용 측정 지표를 산출하기 위한 targets, prediction을 생성
- Training
- 실제 주가와 예측 주가를 비교

### Code
-  https://github.com/hunkim


------------------------------------------
## Team Project 8조
이주완(21500554) / 김강훈(21900063)


### <프로젝트 개요>
1. 학습 알고리즘인 “역전파”에 대해 알아보기
2. “경사하강법”에 대해 알아보기
3. 간단한 코드로 뉴런 네트워크 구현하기
4. 역전파법을 이용한 뉴럴 네트워크를 구현하고, 입력값을 활용해서 결과값을 예측  
### 프로젝트 결과물
- [동영상 설명](https://d1b10bmlvqabco.cloudfront.net/attach/jsflt1hhzmp45n/jldf6qd7zvh6fw/jwhq9jlu8yxv/Group7_Video.mp4)


------------------------------------------
## Emotion/Gender Classification (최은송, 김나연)


### 프로젝트 소개
oarriaga의 emotion/geder classification 프로젝트를 git에서 다운받아 설치하고 모델을 학습해서 실시간 웹캠으로 사람들의 감정 상태와 성별을 탐지하기.

CNN 모델과 openCV를 통해 실시간으로 사람의 얼굴을 탐지하고 감정과 성별을 구별하는 Emotion/Gender Classification 프로젝트

### Original Source
https://github.com/oarriaga/face_classification

### 프로젝트 개요
emotion/gender classification의 원리
oarriaga의 emotion/gender classification 프로젝트를 깃허브에 다운받아서 설치하기
keras와 opencv 환경에서 위에서 다운받은 classification 프로그램 구동하기
사람의 감정과 성별을 구분한 얼굴 사진이 있는 데이터베이스를 classification 모델에 학습시키기
웹캠을 이용해 실시간으로 사람들의 성별과 감정 상태를 탐지하기

### 프로젝트 결과물
- [동영상](https://www.youtube.com/watch?v=aH4svYvwyDA&feature=youtu.be)


------------------------------------------
## 심장질환 예측 (홍유빈, 이수지, 김영민)

### 프로젝트 링크
- [Kaggle Project Link](https://www.kaggle.com/cdabakoglu/heart-disease-classifications-machine-learning/notebook)

### 프로젝트 계기

- 전공과의 연관성을 살려, 생명분야에서 인공지능이 사용되는 사례를 조사하고자 하였다.
지금까지 배웠던 많은 내용들을 포괄할 수 있는 주제를 선정하고자 하였다.

### 프로젝트 개요
1. Heart desease.csv 데이터를 분석한다.
2. Logistic Regression Algorithm을 포함한 다른 Algorithms(KNN , SVM , naive bayes , decision tree, random forest + confusion matrix)을 이용해서 heart disease를 예측해본다.
3. 만든 모델들을 Accuracy와 Confusion Matrix를 이용해서 비교 후 최선의 모델이 무엇인지 찾아낸다.

### 발표자료
- [script.ipynb](https://github.com/idebtor/JoyAI/tree/master/projects/9%EC%A1%B0-%EC%8B%AC%EC%9E%A5%EC%A7%88%ED%99%98(LogisiticRegression))
- [보충자료.docx](https://github.com/idebtor/JoyAI/tree/master/projects/9%EC%A1%B0-%EC%8B%AC%EC%9E%A5%EC%A7%88%ED%99%98(LogisiticRegression))

### 동영상
- [동영상 설명 1](https://youtu.be/ODGJN-cBx7Q)
- [동영상 설명 2](https://youtu.be/cgOMwMOXBX8)


-------------------------------------

## 코스피, 코스닥 증감률에 따른 금값 변화(김석진, 김예준)

### 프로젝트 개요
- 코스피 , 코스닥 증감률의 데이터를 저장
- 금값에 대한 데이터를 저장
- 데이터들을 바탕으로 학습
- 학습된 모델을 활용하여 금값예측

### 프로젝트 결과물
- [동영상 설명](https://youtu.be/WFmorIg0byQ)


--------------------------------------------------

## 파이션을 이용해 Neraul Style Network 구현하기

### 프로젝트 소개 
  - 일반 사진을 유명 화가의 그림같이 보이게 한다

### 프로젝트 개요
- Neural Style Transfer에 대해 알아보기
- 이미지의 스타일을 학습하기
- 학습한 스타일을 바탕으로 Neural Style Transfer 구현하기

# 동영상 
- [이론 설명 동영상](https://www.youtube.com/watch?v=I_IU7AifIXU&feature=youtu.be)
- [구현 설명 동영상](https://www.youtube.com/watch?v=NIlY4Lsx1jk&feature=youtu.be)

-----------------------------
## Tensorflow와 LSTM RNN을 이용하여 아마존 주가 예측하기

### 프로젝트 개요
- 과거와 현재 일별 주가와 거래량을 이용하여 미국 아마존의 내일 주가을 예측한다
- 과거 아마존 주가 데이터를 다운받는다
- LSTM 네트워크 생성, RNN 생성
- 그래프 출력

## 프로젝트 결과물
- [설명 동영상](https://github.com/idebtor/JoyAI/blob/master/projects/13%EC%A1%B0-Amazon%EC%A3%BC%EA%B0%80%EC%98%88%EC%B8%A1/13%EC%A1%B0%20%EC%98%81%EC%83%81.mp4)


--------------------------------------------

## keras와 flask web server를 활용한 타이타닉호 생존자 분석 및 예측 프로그램 (정혜윰, 이원빈)

### 프로젝트 개요
- 1912년 침몰한 타이타닉호 생존 가능성 예측 프로그램입니다.
- 타이타닉호의 탑승자들의 데이터를 받고 정보에 따라 분석해서, keras로 학습합니다. 플라스크 웹 서버에 접속하고 안내를 따라 이름, 성별, 나이, 탑승 호실 등급, 동승자 등의 정보를 입력하면  타이타닉 호 침몰 시 해당 사람의 생존 가능성을 계산하여 알려줍니다.

### github 주소
- https://github.com/hyeyoomj/TitanicSurvivor

### 영상
- [데이터 분석 및 학습하기 튜토리얼](https://youtu.be/NgRdTigDO3o)
- [학습 모델 플라스크 웹 서버와 연동하기 튜토리얼](https://youtu.be/Vl6cupU0MKM)


---------------------------------------------------------
## (Darknet YOLO tensorflow version) "DarkFlow" (최지호, 이재민)

### 프로젝트 개요
- Darket Yolo는 수업시간 ted강의로 나왔던 주제입니다. C와 CUDA로 작성된 오픈 소스 신경망 프레임 워크이며, 빠르고 설치가 쉽고 CPU 및 GPU계산을 지원하고 있습니다.

- 그렇지만 매우 강력한 오픈 소스 신경망 프레임 워크인 Darknet YOLO는 CUDA를 사용하기 때문에 NVIDIA사의 그래픽이 없는 컴퓨터는 CUDA를 사용할 수 없다는 점이 있었고

- 원 저자는 C를 이용하여 프로그램을 짰기 때문에, Tensorflow의 Tensorvboard와 같은 유용한 기능들을 사용할 수 없는 점이 있습니다. 그래서 어느 개발자가 TensorFlow 버전 YOLO인 DarkFlow를 개발하여 오픈소스로 공개하였고 python 기반으로 작성을 했습니다!

### 프로젝트의 목적
- MAC OS에서도 darknet yolo를 구동 할 수 있어 본 프로젝트는 "MAC OS" 환경의 터미널에서 "Darkflow 설치 및 실행해보기" 입니다.

- 프로젝트를 통해
  - Darkflow 개발환경 설치
  - Darkflow 돌려보기
  - 모델 및 여러가지 제공 명령어 학습
  - VOC로 Yolo 학습시키는 법,
  - 객체 탐지 예제 실행해보기(사진 & 실시간 웹캠) with threshold

### 프로젝트의 결과물
  - 동영상: [동영상 설명서](https://www.youtube.com/watch?v=yxQGlMLegcs&feature=youtu.be)
  - 블로그: [포스팅 블로그](https://blog.naver.com/co748/221554688233)
  - ppt파일: [Darknet_Yolo_Tensorflow.pptx](https://github.com/idebtor/JoyAI/blob/master/projects/15%EC%A1%B0/Darknet_Yolo_Tensorflow.pptx)


-------------------------------------------

## AlpaZero 작동하기 (권혁재, 장다빈, 황주영, 류태동)
- 케라스와 파이썬으로 딥마인드 팀의 프로젝트인 알파제로 구동하기

### 프로젝트 개요
- 알파제로의 원리
- 알파고와 알파제로의 차이점
- 케라스 설치 및 구동
- 알파제로를 이용하여 게임을 학습시키기

### 프로젝트 결과물
- [동영상](https://youtu.be/f0GxX9OCrXs)

------------------------------------------
## Tensorflow와 선형회귀를 활용한 날씨에 따른 전력사용량 예측(김수연, 최준혁)

### 프로젝트 아이디어
- 점점 더위가 찾아오면서 누진세등 전기세에 대한 관심도가 높아지고 있는 이때. 기후에 따라 우리는 얼만큼 전력을 사용하는지 알고싶어 전력사용량 예측하는 인공지능을 만들어보면 어떻게 생각해봤다.

### 데이터 수집
- 기상청과 전력통계정보시스템사이트를 통해 기온과전력사용량 데이터를다운받는다

### 데이터 정제
- 우리는 좀더 인식하기 쉽게 수집한 데이터가 일정하지 않는 고객호수등을 감안해서 데이터를 쉽게 적용할 수 있는 형태로 바꾸어 줘야한다.

### 기대효과
1. 월별 전력사용량 예측
2. 기온과 전력사용량에 연관성을 파악할수있다.

-----------------------------------

## 블록체인을 이해하고 간단한 블록체인 코어를 구현해보기(최혁진)

### 프로젝트 소개
- 화폐의 원래 개념과, 은행에서의 transaction vs. 블록체인의 transaction원리를 비교해 보고 블록체인의 코어를 코딩해보고 구동해보는 것입니다.


### 다루게 될 기술
- hashlib
- json
- 파이썬
- class 함수

### 프로젝트 개요
- class를 통해 블록에 거래내역 (nonce, Tstamp, transaction, prevhash)을 부여한다
- hashlib과 json을 import해 블록의 거래내역을 암호화 한다 (sha.256사용)
- class를 이용해 Genesis블록과 블록체인을 구현한다
- for range를 이용해 블록이 손상됬을 경우 블록이 체인에 추가되지 못하도록 한다
- 블록체인 구현을 한것을 임의로 만든 것과, 그 만든 것의 데이터를 조작 (해킹)한 후 블록이 쓰일 수 있는 블록인지 (True) 쓰일 수 없는 블록인지 (Invalid, False) 구분한다


### 동영상
- [Youtube](https://www.youtube.com/watch?v=EA4dXhY91a8&feature=share)

-----------------------------------------

## 파이썬 자연어처리로 영화 리뷰 예측 AI 개발하기(김희수)

### 프로젝트 소개
- 영화 리뷰를 분석하여 긍정적 리뷰인지 부정적 리뷰인지 예측하는 파이썬 자연어처리를 사용한 인공지능을 만들어본다.

### 다루게 될 기술
- NLTK
- pandas
- skikit-learn
- tf-idf
- pickle

### 프로젝트 개요
- kaggle사이트로 영화 리뷰를 다운로드 받고 자료를 살펴보기
- 데이터를 정제하여 단어장 만들기
- countVectorizer()로 단어장을 벡터로 만들기
- tf-idf기법 사용하여 단어의 빈도수 체크하기
- nltk를 사용하여 문장을 단어로 나누기
- 학습 모델 만들기

### 프로젝트 결과물
- [동영상 설명](https://youtu.be/ceCAj_f7yuo)



