# 모두를 위한 인공지능의 활용(2019-1)

# Team Project 1조 - Tensorflow와 선형회귀를 활용하여 silver to gold price 예측 프로젝트.

김정빈 21400179 / 김하림 21400220 / 안재욱 21400422
----------------------------------

##  프로젝트 개요

  플라스크 웹 서버를 통하여 접속할 수 있으며,   은 가격 정보를 입력하면 그에 상응하는 금 가격을 계산하여 출력합니다.   사용자는 이 정보들을 읽고 둘의 상관관계와 투자에 관하여 도움을 받을 수 있습니다.

##  기대 효과

  1. 은을 통해 금의 가격 예측

  2. 금시세와 은값에 의한 예측된 금가격 정보를 비교하여 시장의 흐름 파악에 참고 자료로 활용 가능.

  3. 지난 1년 동안의 daily 데이터셋을 통해 보다 정확한 예측이 가능함.



##  특징

  1. COMEX(뉴욕상품거래소) 기준으로  달러/트로이온스(US dollar per troy ounce) 단위에 따른 정보 예측.

  2. 다중 선형 회귀가 아닌 단순 선형 회귀 활용으로 그에 따른 코드의 쓰임을 파악하는데 도움.


## [금 은 퀀트 투자 AI 프로젝트 영상](https://www.youtube.com/watch?v=Vbd7WszNr4U&feature=youtu.be)

## 결과물
  silver_gold_data.csv

  silver_gold_predict_01.ipynb

  silver_gold_predict_02.ipynb

  금_은_퀀트_투자_AI_프로젝트.pptx

  1조_프로젝트.zip


------------------------------------------
# Team Project 9조

홍유빈(21500804) / 이수지(21800528) / 김영민(21900141)

## <프로젝트 링크>

[Kaggle Project Link](https://www.kaggle.com/cdabakoglu/heart-disease-classifications-machine-learning/notebook)

## <프로젝트 계기>

전공과의 연관성을 살려, 생명분야에서 인공지능이 사용되는 사례를 조사하고자 하였다.
지금까지 배웠던 많은 내용들을 포괄할 수 있는 주제를 선정하고자 하였다.

## <프로젝트 개요>

1. Heart desease.csv 데이터를 분석한다.
2. Logistic Regression Algorithm을 포함한 다른 Algorithms(KNN , SVM , naive bayes , decision tree, random forest + confusion matrix)을 이용해서 heart disease를 예측해본다.
3. 만든 모델들을 Accuracy와 Confusion Matrix를 이용해서 비교 후 최선의 모델이 무엇인지 찾아낸다.

## <발표자료>

[script.ipynb](https://github.com/idebtor/JoyAI/tree/master/projects/9%EC%A1%B0-%EC%8B%AC%EC%9E%A5%EC%A7%88%ED%99%98(LogisiticRegression))

[보충자료.docx](https://github.com/idebtor/JoyAI/tree/master/projects/9%EC%A1%B0-%EC%8B%AC%EC%9E%A5%EC%A7%88%ED%99%98(LogisiticRegression))

* 동영상은 나누어 제작(1편, 2편)
[동영상 설명 1](https://youtu.be/ODGJN-cBx7Q)
[동영상 설명 1](https://youtu.be/cgOMwMOXBX8)

---------------------------------

# Team Project 18조 - 블록체인을 이해하고 간단한 블록체인 코어를 구현해보기

최혁진(21900771)
-------------------------------

## <프로젝트 소개>
화폐의 원래 개념과, 은행에서의 transaction vs. 블록체인의 transaction원리를 비교해 보고 블록체인의 코어를 코딩해보고 구동해보는 것입니다.



## <다루게 될 기술>
hashlib
json
파이썬
class 함수

## <프로젝트 개요>
class를 통해 블록에 거래내역 (nonce, Tstamp, transaction, prevhash)을 부여한다

hashlib과 json을 import해 블록의 거래내역을 암호화 한다 (sha.256사용)

class를 이용해 Genesis블록과 블록체인을 구현한다

for range를 이용해 블록이 손상됬을 경우 블록이 체인에 추가되지 못하도록 한다

블록체인 구현을 한것을 임의로 만든 것과, 그 만든 것의 데이터를 조작 (해킹)한 후 블록이 쓰일 수 있는 블록인지 (True) 쓰일 수 없는 블록인지 (Invalid, False) 구분한다


<동영상 링크>
https://www.youtube.com/watch?v=EA4dXhY91a8&feature=share

21900771_최혁진_Block_Chain_.ipynb

21900771_최혁진__미래의_화폐로서의_가능성_블록체인.pptx

--------------------------------------------------

# Team Project 12조 - 파이션을 이용해 Neraul Style Network 구현하기

## < 프로젝트 소개 >

  - 일반 사진을 유명 화가의 그림같이 보이게 한다
--------------------------------------------------------

## <프로젝트 개요>
- Neural Style Transfer에 대해 알아보기

- 이미지의 스타일을 학습하기

- 학습한 스타일을 바탕으로 Neural Style Transfer 구현하기

## <프로젝트 산출물>
piazza_clz_md_2009_piazza_clz_md_2009

CoreAAC_DirectShow_Filter.exe

StyleTranfer_소개.docx

Neural_Style_Transfer_with_Eager_Execution.ipynb

[이론 설명 동영상](https://www.youtube.com/watch?v=I_IU7AifIXU&feature=youtu.be)
[구현 설명 동영상](https://www.youtube.com/watch?v=NIlY4Lsx1jk&feature=youtu.be)

(Good sound track video in YouTube)


--------------------------------------------

# Team Project 14 - keras와 flask web server를 활용한 타이타닉호 생존자 분석 및 예측 프로그램

정혜윰(21700680)/이원빈(21900608)
---------------------------
## 프로젝트 개요

1912년 침몰한 타이타닉호 생존 가능성 예측 프로그램입니다.

타이타닉호의 탑승자들의 데이터를 받고 정보에 따라 분석해서, keras로 학습합니다. 플라스크 웹 서버에 접속하고 안내를 따라 이름, 성별, 나이, 탑승 호실 등급, 동승자 등의 정보를 입력하면  타이타닉 호 침몰 시 해당 사람의 생존 가능성을 계산하여 알려줍니다.

## [github 주소](https://github.com/hyeyoomj/TitanicSurvivo)
https://github.com/hyeyoomj/TitanicSurvivor

## Video 1
[데이터 분석 및 학습하기 튜토리얼](https://youtu.be/NgRdTigDO3o)
https://www.youtube.com/watch?v=NgRdTigDO3o&feature=youtu.be

## Video 2
[학습 모델 플라스크 웹 서버와 연동하기 튜토리얼](https://www.youtube.com/watch?v=Vl6cupU0MKM)
https://youtu.be/Vl6cupU0MKM

## 자료 및 코드파일
TitanicSurvivormaster.zip

----------------------------------------------------
# Team Project 4조 - MCTS를 활용한 인공지능 TicTacToe 개발

장재근(21400659) / 이화평(21800608)
--------------------------------------------
### < 프로젝트 소개 >
- 몬테카를로 트리 탐색(MCTS) 및 UCT 알고리즘을 이용한 인공지능 TICTACTOE게임 프로그램 제작

### <다루게 될 기술>
- Python

### < 프로젝트 개요 >
- 몬테 카를로 트리 탐색에 대한 개념 이해
- 파이썬을 이용하여 Tic-Tac-Toe 게임 구현
- 파이썬을 이용하여 몬테카를로 트리 탐색 코드 구현
- 파이썬을 이용하여 UCT (Upper Confidence Bounds of Tree) 알고리즘 구현
- 게임 실행 및 확인

### < 동영상 링크 >
1) 몬테카를로 및 몬테카를로 트리 서치에 대한 이해

 https://www.youtube.com/watch?v=6-cjW6X9HAI

2) MCTS 및 UCT 알고리즘 직접적 구현 및 시연
 https://www.youtube.com/watch?v=kSZ0Ovu7nug

3) 발표 자료 및 구현 코드 (.py & .ipynb)

TICTACTOE_MCTS.zip

인공지능의_활용_ppt.pptx

------------------------------------------
# Team Project 11조 - 코스피, 코스닥 증감률에 따른 금값 변화

김석진 21700105 김예준 21900156
-----------------------------------
## <프로젝트 개요>
- 코스피 , 코스닥 증감률의 데이터를 저장
- 금값에 대한 데이터를 저장
- 데이터들을 바탕으로 학습
- 학습된 모델을 활용하여 금값예측

## <프로젝트 결과물>
https://youtu.be/WFmorIg0byQ

18조_프로젝트.pptx

goldPrice.csv

goldPrice_Predict.ipynb

goldPricePrecit2.ipynb

------------------------------------------
# Team Project 16/19조 - AlpaZero 작동하기

16조 21800041 권혁재 / 21900625 장다빈, 19조 21900819 황주영 / 21800235 류태동

-----------------------------------

케라스와 파이썬으로 딥마인드 팀의 프로젝트인 알파제로 구동하기

## <프로젝트 개요>

- 알파제로의 원리
- 알파고와 알파제로의 차이점
- 케라스 설치 및 구동
- 알파제로를 이용하여 게임을 학습시키기

## <프로젝트 결과물>
https://youtu.be/f0GxX9OCrXs

------------------------------------------
# Team Project 6조 - 다음날 주가 예측하기

 홍순찬(21400817)

 -------------------------------
## <프로젝트 소개>
RNN 알고리즘을 사용하여 과거 주가데이터 학습한 후, 다음날 주가 예측하기

## <다루게 될 기술>

numpy
tensorflow
matplotlib
os

## <프로젝트 개요>

과거의 주가 데이터를 학습
학습용/ 테스트용 데이터를 사용
검증용 측정 지표를 산출하기 위한 targets, prediction을 생성
Training
실제 주가와 예측 주가를 비교

## 프로젝트 결과물
- RNN_과거_주가_데이터_학습하여_다음날_종가_예측하기.pptx

- 과거_주가_데이터_학습하여_다음날_종가_예측하기.ipynb

- stock_daily_price.csv

-----------------------------------

---------------------------------------------------------
# Team Project 15조 : (Darknet YOLO tensorflow version) "DarkFlow"
21700748 최지호, 21900556 이재민

## 프로젝트 개요
____________________________________________
## Darknet YOLO tensorflow version "DarkFlow"에 대하여
--------
+ Darket Yolo는 수업시간 ted강의로 나왔던 주제입니다. C와 CUDA로 작성된 오픈 소스 신경망 프레임 워크이며, 빠르고 설치가 쉽고 CPU 및 GPU계산을 지원하고 있습니다.

+ 그렇지만 매우 강력한 오픈 소스 신경망 프레임 워크인 Darknet YOLO는 CUDA를 사용하기 때문에 NVIDIA사의 그래픽이 없는 컴퓨터는 CUDA를 사용할 수 없다는 점이 있었고

+ 원 저자는 C를 이용하여 프로그램을 짰기 때문에, Tensorflow의 Tensorvboard와 같은 유용한 기능들을 사용할 수 없는 점이 있습니다. 그래서 어느 개발자가 TensorFlow 버전 YOLO인 DarkFlow를 개발하여 오픈소스로 공개하였고 python 기반으로 작성을 했습니다!

------

## 프로젝트의 목적
MAC OS에서도 darknet yolo를 구동 할 수 있어 본 프로젝트는 "MAC OS" 환경의 터미널에서 "Darkflow 설치 및 실행해보기" 입니다.

+ 프로젝트를 통해

 1. Darkflow 개발환경 설치
 2. Darkflow 돌려보기
 3. 모델 및 여러가지 제공 명령어 학습
 4. VOC로 Yolo 학습시키는 법,
 5. 객체 탐지 예제 실행해보기(사진 & 실시간 웹캠) with threshold

을 해보았습니다.

-----------------------------

  동영상: [동영상 설명서](https://www.youtube.com/watch?v=yxQGlMLegcs&feature=youtu.be)

  블로그: [포스팅 블로그](https://blog.naver.com/co748/221554688233)

  ppt파일: [Darknet_Yolo_Tensorflow.pptx](https://github.com/idebtor/JoyAI/blob/master/projects/15%EC%A1%B0/Darknet_Yolo_Tensorflow.pptx)

  코드: 해당 블로그 주소 포스팅을 통해 git clone이 가능합니다.




-----------------------------------
