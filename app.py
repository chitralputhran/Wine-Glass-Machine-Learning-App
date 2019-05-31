from flask import Flask, render_template, url_for, request

import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    model_1 = pickle.load(open('data/model.sav', 'rb'))
    if request.method == 'POST':
        fa = request.form['faInputName']
        va = request.form['vaInputName']
        ca = request.form['caInputName']
        rs = request.form['rsInputName']
        ch = request.form['chInputName']
        fs = request.form['fsInputName']
        ts = request.form['tsInputName']
        ds = request.form['dsInputName']
        pH = request.form['pHInputName']
        sp = request.form['spInputName']
        al = request.form['alInputName']

    form_data = [fa, va, ca, rs, ch, fs, ts, ds, pH, sp, al]
    num_data = []
    for element in form_data:
        num_data.append(float(element))
    predict_this = np.array(num_data).reshape(1, -1)
    y_pred = model_1.predict(predict_this)
    y_prob = model_1.predict_proba(predict_this)
    prob = y_prob[:, 1]

    return render_template('result.html', y_pred=y_pred, y_prob=prob, fa=fa, va=va, ca=ca, rs=rs, ch=ch, fs=fs, ts=ts, ds=ds, pH=pH, sp=sp, al=al)


if __name__ == '__main__':
    app.run()
