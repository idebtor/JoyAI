# file name : index.py
# pwd : /project_name/app/main/index.py
 
from flask import Blueprint, request, render_template, flash, redirect, url_for
from flask import current_app as app
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns

main= Blueprint('main', __name__, url_prefix='/')
file_name = '';
price = pd.DataFrame()
x_train = np.ndarray([])
x_test = np.ndarray([])
y_train = np.ndarray([])
y_test = np.ndarray([])
scaler = '';

@main.route('/', methods=['GET'])
def index():

    # /main/index.html은 사실 /project_name/app/templates/main/index.html을 가리킵니다.
    return render_template('/main/index.html', data="Hello")
    
@main.route('/getFile', methods=['GET', 'POST'])
def getFile():
    if request.method == 'POST':
        value = request.form['file']
        file_name = str(value)
    #csv path 를 아래 안에다 적어주시면 됩니다!
    
    import os

    data = pd.read_csv(os.getcwd()+'/'+file_name)
    #data = pd.read_csv(os.getcwd()+'/LG.csv')
    global price
    price  = data[['close']]
    
    global scaler

    scaler = MinMaxScaler(feature_range = (-1,1))
    price['close'] =scaler.fit_transform(price['close'].values.reshape(-1,1))
    
    
    return render_template('/main/getFile.html', name=file_name, data=price['close'])
    

@main.route('/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        value = request.form['test']
        value = int(value) * 0.1

    lookback = 20 # choose sequence length
    x_train, y_train, x_test, y_test = split_data(value, price, lookback)
    

    return render_template('/main/split.html', xt = x_train.shape, yt = y_train.shape, xe = x_test.shape, ye = y_test.shape)

def split_data(value, stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    
    # 8:2 split (train, test)
    test_set_size = int(np.round(value *data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    global x_train
    global y_train
    global x_test
    global y_test
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

@main.route('/learn', methods=['GET', 'POST'])
def learn():
    if request.method == 'POST':
        v1 = request.form['input_dim']
        v1 = int(v1)
        v2 = request.form['hidden_dim']
        v2 = int(v2)
        v3 = request.form['num_layers']
        v3 = int(v3)
        v4 = request.form['output_dim']
        v4 = int(v4)
        v5 = request.form['num_epochs']
        v5 = int(v5)
        
    
    global x_train
    global y_train
    global x_test
    global y_test
    
    #input features
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    # y output
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)


    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)
    
    input_dim = v1
    hidden_dim = v2
    num_layers = v3
    output_dim = v4
    num_epochs = v5
    
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    #MSE metric
    criterion = torch.nn.MSELoss(reduction='mean')
    #Adam optimiser # learning rate =  0.01
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    
    
    import time

    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))
    
    predict = pd.DataFrame(y_train_pred.detach().numpy())
    original = pd.DataFrame(y_train_lstm.detach().numpy())

    #그래프
    legend = 'Data'
    legend2 = 'Training Prediction (LSTM)'
    legend3 = 'Loss'
    
    labels = [original.index.tolist()][0]
    values = original[0].tolist()
    
    values2 = predict[0].tolist()
    
    labels_h = [i for i in range(len(hist))]
    
    import math, time
    from sklearn.metrics import mean_squared_error

    global scaler
    # make predictions
    y_test_pred = model(x_test)

    # inverse scaling predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    lstm.append(trainScore)
    lstm.append(testScore)
    lstm.append(training_time)

    y_label = [y for y in range(len(y_test))]
    
    y_test_pred = np.ravel(y_test_pred, order='C')
    y_test = np.ravel(y_test, order='C')
    
    return render_template('/main/learn.html',
    legend=legend, labels=labels, values=values, legend2=legend2, values2 = values2, hist = hist, labels_h = labels_h, legend3=legend3, trainScore=trainScore, testScore = testScore, y_label = y_label, y_test_pred = y_test_pred, y_test = y_test
    )

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
