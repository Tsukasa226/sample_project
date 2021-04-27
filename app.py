from types import MethodDescriptorType
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import csv
import os
from sklearn.ensemble import RandomForestRegressor 
import pickle

app = Flask(__name__)

# フォーム表示
@app.route('/', methods=['GET'])
def index():
    return render_template("home.html")
    

# アップロード機能
@app.route('/upload', methods=['POST'])
def upload():
    try:
        print(request.files)
      
        fs = request.files['file']
        
        fs.save(fs.filename)
        # アップしたファイルをインサートする
        reading_csv(fs.filename)

        return render_template("uploaded.html")
    except:
        return "ファイルが適切ではありません"


def reading_csv(filename):

    data = pd.read_csv(filename)

    strdata_list = str_list()

    for colum_name in strdata_list:
        data = data.drop([colum_name] , axis=1)

    test_x_worknum = data['お仕事No.']
    data = data.drop(['お仕事No.'], axis=1)
    data = data.fillna(0)#0で補完

    rfr = model_create(strdata_list)

    out_y = rfr.predict(data)
    out_y = pd.DataFrame(out_y, columns=['応募数 合計'])

    OUT_data = pd.concat([test_x_worknum, out_y], axis=1)
    OUT_data.to_csv('files/submit_datas.csv', index=False, encoding='utf_8_sig')

def str_list():
    df = pd.read_csv('train_x.csv')

    strdata_list = []
    for column in df.columns:
        if df[column].dtype == object:
            strdata_list.append(column)
        else:
            if str(df[column].describe()['mean']) == 'nan':
                strdata_list.append(column)

    return strdata_list

def data_all(strdata_list):
    
    df = pd.read_csv('train_x.csv')

    for data in strdata_list:
        df = df.drop([data] , axis=1)

    df = df.fillna(0)#0で補完
    df= df.drop_duplicates()

    train_y = pd.read_csv('train_y.csv')
    train_y = train_y.drop_duplicates()

    train_y_indexs = train_y["お仕事No."].drop_duplicates()
    applys = []
    for index in train_y_indexs:
        apply = train_y[train_y['お仕事No.']==index].sum()
        applys.append(apply)

    applys = pd.DataFrame(applys)

    df_all = pd.merge(df, applys, on='お仕事No.', how='inner')

    return df_all


def model_create(strdata_list):

    df_all = data_all(strdata_list)

    X = np.array(df_all.drop(['お仕事No.', '応募数 合計'], axis=1))
    y = np.array(df_all['応募数 合計'])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    rfr = RandomForestRegressor(random_state=0)
    #推論
    rfr.fit(X, y)
    # y_pred = rfr.predict(X_test)

    # print('二乗誤差平均：',np.sqrt(mean_squared_error(y_pred, y_test)))

    return rfr

@app.route('/down_load', methods=['GET'])

def download_file():
    return send_file("files/submit_datas.csv", as_attachment=True)

if __name__ == '__main__':
    app.run()