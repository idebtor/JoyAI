# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import numpy as np
import keras
import tensorflow as tf
import flask

# flask model 초기화
app = Flask(__name__)

# 모델 불러오기
from keras.models import load_model
model = load_model('./model/saved_model.h5')
graph = tf.get_default_graph()
# model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # 파라미터 받기
        name = str(request.form['name'])
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        pclass = float(request.form['pclass'])
        sibling = float(request.form['sibling'])
        parent = float(request.form['parent'])
        fare = float(request.form['fare'])

    with graph.as_default():
        # 변수 선언
        survive = 0

        # 입력된 파라미터를 배열 형태로
        data = np.array([pclass, sex, age, sibling, parent, fare]).reshape(1,6)
        dict = model.predict(data)
        survive = dict[0]*100

        return render_template('index.html', name=name, survive=survive)

if __name__ == '__main__':
   app.run(debug = True)
