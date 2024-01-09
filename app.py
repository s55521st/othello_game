from flask import Flask, render_template, request
import funtion

import sys
import random
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


app = Flask(__name__)

# オセロ盤を初期化
board = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 1, 0, 0, 0],
    [0, 0, 0, 1, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

current_player = 1
current_board = board
retry_num =0


class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
                l1=L.Linear(64, 100),
                l2=L.Linear(100, 100),
                l3=L.Linear(100, 64),
        )
 
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
    
class Classifier(Chain):#ネットワークのトレーニングおよび評価
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):# t 正解ラベル
        y = self.predictor(x)# xをpredictorに渡し出力を得る 
        loss = F.softmax_cross_entropy(y, t)#交差エントロピー誤差（softmax_cross_entropy）損失
        accuracy = F.accuracy(y, t)#精度
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss
    

#モデルの読み込み
model_1 = Classifier(MLP())
serializers.load_npz('model/model_ggs_white_64.npz', model_1)


def make_cpu_move(board):
    # モデルに入力するデータを準備
    global retry_num, current_player
    input_data = np.array([board], dtype=np.float32)  # 入力データを適切な形に整形
    # Chainerを使って予測を行う
    with chainer.using_config('train', False):
        prediction = model_1.predictor(input_data)
        print(prediction)
    
    # 予測結果から、CPUの手を決定
    y1 = F.softmax(model_1.predictor(input_data))  
    sorted_arg = np.argsort(y1.data)[0][::-1]
    n = sorted_arg[retry_num]
        
    

    # ゲームのルールに従って石を置く処理
    if funtion.is_valid_move(board, int(n / 8), int(n % 8), 2):
        funtion.make_move(board, int(n / 8), int(n % 8), 2)
        retry_num = 0
        current_player = 1 if current_player == 2 else 2
    else:
        retry_num += 1
        print('retry_num{}'.format(retry_num))
        make_cpu_move(board)
    
    

# ゲームの進行
@app.route('/move', methods=['GET', 'POST'])
def game():
    global current_player, current_board  # グローバル変数として現在のプレイヤーと盤面を使用

    if request.method == 'POST':
        current_board = np.array(current_board)
        row = int(request.form['row'])
        col = int(request.form['col'])

        if current_player == 1:
            # ゲームのルールに従って石を置く処理（人間が手を打つ場合）
            if funtion.is_valid_move(current_board, row, col, 1):
                funtion.make_move(current_board, row, col, 1)
            else:
                warning = 'そこに石を置けません'
                return render_template('othello.html', board=current_board, current_player=current_player, warning=warning)

        # プレイヤーを切り替える
        current_player = 2 if current_player == 1 else 1
        
        # CPUの手を決定して石を置く処理（プレイヤーが'2'の場合）
        if current_player == 2:
            make_cpu_move(current_board)

    return render_template('othello.html', board=current_board, current_player=current_player)

