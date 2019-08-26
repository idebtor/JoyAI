# Titanic Survivor Project
### 타이타닉호의 생존자 데이터 분석 및 keras 를 활용한 생존율 예측 프로그램

![image](./data/titanic.jpeg)
by 정혜윰(21700680) 이원빈(21900000), 모두를 위한 인공지능 활용, HGU

## 프로젝트 개요
1912년 침몰한 타이타닉호 생존 가능성 예측 프로그램입니다. 타이타닉호의 탑승자들의 데이터를 받고 정보에 따라 분석해서, keras로 학습합니다.<br>
또한, 학습모델을 플라스크 웹 서버와 연동하도록 구현하여 서버에 접속하고 안내를 따라 사용자의 정보를 입력하면 타이타닉 호 침몰 시 해당 사용자의 생존 가능성을 계산하여 알려주는 프로그램입니다.

## 사용한 도구
> 사용한 언어: python<br>
사용한 라이브러리: 학습을 위한 keras, 데이터 분석 및 시각화를 위한 matplotlib, 파일을 읽어오기 위한 pandas, 그밖의 numpy 등<br>
데이터 출처: kaggle (https://www.kaggle.com/broaniki/titanic)


## 학습 모델 레이어 구성도
![image](./data/train_model.jpeg)

## Demo Web Server
![image](./data/demoserver.png)

## 파이썬 데이터 학습 모델 생성
```
# 깃헙에서 소스코드를 다운로드 받습니다.
git clone https://github.com/hyeyoomj/TitanicSurvivor.git

# 프로젝트 폴더로 이동합니다.
cd TitanicSurvivor

# 주피터노트북을 실행해 접속합니다.
jupyter notebook

# data_train.ipynb 프로그램을 실행시켜 학습을 후 모델을 저장합니다.
```

## 서버 실행 명령어
```
# 프로젝트 폴더로 이동합니다.
cd TitanicSurvivor

# 학습 후 저장되어있는 모델 파일을 WebServer>model 폴더로 이동시킵니다.
mv saved_model.h5 ./WebServer/model

# 플라스크 웹 서버 폴더로 이동합니다.
cd WebServer

# 웹 서버를 실행합니다.
python server.py

# 실행되면 인터넷 주소창에 "localhost:5000" 를 입력해 웹 서버에 접속합니다.
```


## 데이터 통계 분석
```
# 프로젝트 폴더로 이동합니다.
cd TitanicSurvivor

# 주피터노트북을 실행해 접속합니다.
jupyter notebook

# data_analysis.ipynb 프로그램을 실행시킵니다.
```

copyright by 정혜윰, 이원빈
