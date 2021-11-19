# Sentiment Analysis
#### Sentiment Analysis of Youtube Comments

[presentation](https://youtu.be/_1WixhS6dyg) | [pdf](https://github.com/dodoyoon/SentimentAnalysis/blob/main/presentation.pdf)

### How to run this web server
1. Download data input <br>
Download data of twitter dataset with sentiment from this [link](https://www.kaggle.com/kazanova/sentiment140)<br>
Place this data in an input folder in root directory

2. Build model <br>
Open <b>model.ipynb</b> and execute all cells <br>
<b>model.h5</b> and <b>tokenizer.pkl</b> will be created <br>

3. Run Web Server <br>
~~~
python3 app.py
~~~

4. Open Web Server <br> 
Open webserver by entering address below on browser
~~~
127.0.0.1:5000
~~~

5. Analyse <br>
Enter youtube url on the web server page. The program will automatically collect comments from that youtube video and perform sentiment analysis on them. The results will be shown as pie chart. 

### Reference

Flask referenced from this [github](https://github.com/vyashemang/flask-salary-predictor) <br>
The method to build model was referenced from this [link](https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis) <br>
