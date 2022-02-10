# Baseball hitter's Salary Prediction
### Team 4 : Juna Lim,  Yoon Suk Lee, Gi Wook Lee

## Why

- 류현진, KBO에서의 마지막 리그, 2012년 한화에서의 연봉은 4억 3천만원
   (Korean Baseball Player Hyun-Jin Ryu's, the last season's salary in Korean League, was 430 million won)
- 스포츠에서는 어떻게 연봉이 책정될까?
   (How the salary of baseball player would be set?)
- 정말 능력 위주로 연봉이 책정될까?
   (Is that Really set based on seasonal stats of baseball player?)
   
## Goal

 - Feature : 야구 통계 사이트 statiz에 기재되어 있는 타자의 데이터를 기반으로(Based on annual hitter's statistics in statiz.co.kr)
 
 - Target :  내년 연봉을 예측( predict a next season's annual salary)
 
## Preparation
 
 - Data collection
  - statiz 웹사이트에서 타자의 연봉(Y) 및 기록(X)를 크롤링(Crawled hitter's basic stats & salary information in statiz's website)
  - crawl data at (http://www.statiz.co.kr/stat.php)
 
 - Analysis Method
  - Regression
   - 가중평균을 이용한 회귀분석(Regression by weighted average method)
   - PCA를 이용한 주성분 회귀분석(Principal component regression analysis by PCA)

## Task Plan

### 데이터수집 > 전처리 > 모델 선택 > 계수추정 > 평가 > 개선 > 최종 성능 평가

# Prerequisite

- python 2.7
- jupyter notebook
