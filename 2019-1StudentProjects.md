# 모두를 위한 인공지능의 활용(2019-1)

----------------------------------------------------------
# Team Project _ 15조 : (Darknet YOLO tensorflow version) "DarkFlow"
21700748 최지호, 21900556 이재민

## 프로젝트 개요
____________________________________________
## Darknet YOLO tensorflow version "DarkFlow"에 대하여
--------
+ Darket Yolo는 수업시간 ted강의로 나왔던 주제입니다. C와 CUDA로 작성된 오픈 소스 신경망 프레임 워크이며, 빠르고 설치가 쉽고 CPU 및 GPU계산을 지원하고 있습니다.

+ 그렇지만 매우 강력한 오픈 소스 신경망 프레임 워크인 Darknet YOLO는 CUDA를 사용하기 때문에 NVIDIA사의 그래픽이 없는 컴퓨터는 CUDA를 사용할 수 없다는 점이 있었고

+ 원 저자는 C를 이용하여 프로그램을 짰기 때문에, Tensorflow의 Tensorvboard와 같은 유용한 기능들을 사용할 수 없는 점이 있습니다. 그래서 어느 개발자가 TensorFlow 버전 YOLO인 DarkFlow를 개발하여 오픈소스로 공개하였고 python 기반으로 작성을 했습니다!

따라서
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

## 2장 [경사하강, 신경망이 학습하는 방법](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2)
  - 3Blue1Brown
  - 21:01 (조회 1.7 million)
  - 경사하강법에 사용하는 비용함수에 대한 미분과 그 이유,경사하강법(Gradient Descent)의 작동 원리를 설명하고, 이를 이용하여 다층 인공신경망이 어떻게 학습을 하는지 설명합니다.  
  의 원리를 쉽게 설명합니다. 특히 기계학습의 기초를 이야기할 때마다 사용하는 MNIST(엠니스트) Dataset를 사용하여 숫자를 분별하는 원리를 설명합니다.
  - 영어 + 한국어 자막

## 3장 Optional [역전파 What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U&index=3&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  - Speaker: 3Blue1Brown
  - 13:54 (조회 1.1 million)
  인공신경망 알고리즘의 핵심은 역전파(backpropagation) 알고리즘인데, 이 역전파의 원리를 탁월한 시각적 효과를 사용하여 설명합니다.
  의 원리를 쉽게 설명합니다. 특히 기계학습의 기초를 이야기할 때마다 사용하는 MNIST(엠니스트) Dataset를 사용하여 숫자를 분별하는 원리를 설명합니다.
  - 영어 + 영어 자막

## 4장 Optional [backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)
- Speaker: 3Blue1Brown
- 10:17 (조회 0.7 million)
- This one is a bit more symbol heavy, and that's actually the point.  The goal here is to represent in somewhat more formal terms the intuition for how backpropagation works in part 3 of the series, hopefully providing some connection between that video and other texts/code that you come across later.
- 영어 + 영어 자막

## Optional [심화 학습과 텐서플로루 특강 Tensorflow and deep learning](https://www.youtube.com/watch?v=vq2nnJ4g6N0)
  - Speaker: Devoxx
  - 2:35:52
  - 영어 + 영어 자막

  _One thing I know, I was blind but now I see. John 9:25_
