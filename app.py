from flask import Flask, render_template, request,session
import function

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


# オセロ盤を初期化 1:黒　2：白
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
    




def make_cpu_move(board,model):
    # モデルに入力するデータを準備
    global retry_num, current_player
    input_data = np.array([board], dtype=np.float32)  # 入力データを適切な形に整形
    # Chainerを使って予測を行う
    with chainer.using_config('train', False):
        prediction = model.predictor(input_data)
        print(prediction)
    
    # 予測結果から、CPUの手を決定
    y1 = F.softmax(model_1.predictor(input_data))  
    sorted_arg = np.argsort(y1.data)[0][::-1]
    n = sorted_arg[retry_num]
        
    

    # ゲームのルールに従って石を置く処理
    if function.is_valid_move(board, int(n / 8), int(n % 8), 2):
        function.make_move(board, int(n / 8), int(n % 8), 2)
        retry_num = 0
        current_player = 1 if current_player == 2 else 2
    else:
        retry_num += 1
        print('retry_num{}'.format(retry_num))
        make_cpu_move(board)


def make_cpu_move_black(board,model):
    # モデルに入力するデータを準備
    global retry_num, current_player
    input_data = np.array([board], dtype=np.float32)  # 入力データを適切な形に整形
    # Chainerを使って予測を行う
    with chainer.using_config('train', False):
        prediction = model.predictor(input_data)
        print(prediction)
    
    # 予測結果から、CPUの手を決定
    y1 = F.softmax(model_1.predictor(input_data))  
    sorted_arg = np.argsort(y1.data)[0][::-1]
    n = sorted_arg[retry_num]
        
    

    # ゲームのルールに従って石を置く処理
    if function.is_valid_move(board, int(n / 8), int(n % 8), 1):
        function.make_move(board, int(n / 8), int(n % 8), 1)
        retry_num = 0
        current_player = 2 if current_player == 1 else 1
    else:
        retry_num += 1
        print('retry_num{}'.format(retry_num))
        make_cpu_move_black(board,model)


@app.route('/')#デコレーター
def index():#指定したパスにアクセスした際に実行される関数
    print('access')
    global current_board,current_player,retry_num 
    retry_num = 0
    current_player = 1
    current_board = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    return render_template('model_select.html')

@app.route('/move')#デコレーター
def move():#指定したパスにアクセスした際に実行される関数
    return render_template('model_select.html')


#モデルの読み込み
model_1 = Classifier(MLP())
serializers.load_npz('model/model_ggs_white_64.npz', model_1)

model_2 = Classifier(MLP())
serializers.load_npz('model/model_flance_white.npz', model_2)

model_3 = Classifier(MLP())
serializers.load_npz('model/model_ggs_black_64.npz', model_3)

model_4 = Classifier(MLP())
serializers.load_npz('model/model_flance_black.npz', model_4)




# ゲームの進行
@app.route('/model1', methods=['GET', 'POST'])
def game_model1():

    global current_player, current_board ,model_1  # グローバル変数として現在のプレイヤーと盤面を使用

    num_1 = ' YOU'
    num_2 = ' CPU(GGS_後手モデル)'
    move_action = "/model1"

    if request.method == 'POST':
        current_board = np.array(current_board)
        if(request.form['row']=='' or request.form['col'] == ''):
            return render_template('othello.html', board=current_board, current_player=current_player,num_1=num_1,num_2=num_2)
        else:
            row = int(request.form['row'])
            col = int(request.form['col'])

        

        if current_player == 1:
            # ゲームのルールに従って石を置く処理（人間が手を打つ場合）
            if function.is_valid_move(current_board, row, col, 1):
                function.make_move(current_board, row, col, 1)
            else:
                warning = 'そこに石を置けません'
                return render_template('othello.html', board=current_board, current_player=current_player, warning=warning,num_1=num_1,num_2=num_2)

        # プレイヤーを切り替える
        current_player = 2 if current_player == 1 else 1
        
        # CPUの手を決定して石を置く処理（プレイヤーが'2'の場合）
        if current_player == 2:
            make_cpu_move(current_board,model_1)

    return render_template('othello.html', board=current_board, current_player=current_player,move_action=move_action,num_1=num_1,num_2=num_2)


@app.route('/model2', methods=['GET', 'POST'])
def game_model2():
    
    global current_player, current_board,mod , model_2  # グローバル変数として現在のプレイヤーと盤面を使用

    num_1 = ' YOU'
    num_2 = ' CPU(France_後手モデル)'
    move_action = "/model2"

    if request.method == 'POST':
        current_board = np.array(current_board)
        if(request.form['row']=='' or request.form['col'] == ''):
            return render_template('othello.html', board=current_board, current_player=current_player,num_1=num_1,num_2=num_2)
        else:
            row = int(request.form['row'])
            col = int(request.form['col'])

        if current_player == 1:
            # ゲームのルールに従って石を置く処理（人間が手を打つ場合）
            if function.is_valid_move(current_board, row, col, 1):
                function.make_move(current_board, row, col, 1)
            else:
                warning = 'そこに石を置けません'
                return render_template('othello.html', board=current_board, current_player=current_player, warning=warning,num_1=num_1,num_2=num_2)

        # プレイヤーを切り替える
        current_player = 2 if current_player == 1 else 1
        
        # CPUの手を決定して石を置く処理（プレイヤーが'2'の場合）
        if current_player == 2:
            make_cpu_move(current_board,model_2)

    return render_template('othello.html', board=current_board, current_player=current_player,move_action=move_action, num_1=num_1,num_2=num_2)



@app.route('/model3', methods=['GET', 'POST'])
def game_model3():
    
    global current_player, current_board , model_3 # グローバル変数として現在のプレイヤーと盤面を使用

    num_2 = ' YOU'
    num_1 = ' CPU(GGS_先手モデル)'
    move_action = "/model3"

    if request.method == 'GET':
        make_cpu_move_black(current_board,model_3)


    if request.method == 'POST':
        current_board = np.array(current_board)
        if(request.form['row']=='' or request.form['col'] == ''):
            return render_template('othello.html', board=current_board, current_player=current_player,num_1=num_1,num_2=num_2)
        else:
            row = int(request.form['row'])
            col = int(request.form['col'])

        if current_player == 2:
            # ゲームのルールに従って石を置く処理（人間が手を打つ場合）
            if function.is_valid_move(current_board, row, col, 2):
                function.make_move(current_board, row, col, 2)
            else:
                warning = 'そこに石を置けません'
                return render_template('othello.html', board=current_board, current_player=current_player, warning=warning,num_1=num_1,num_2=num_2)

        # プレイヤーを切り替える
        current_player = 1 if current_player == 2 else 2
        
        # CPUの手を決定して石を置く処理（プレイヤーが'2'の場合）
        if current_player == 1:
            make_cpu_move_black(current_board,model_3)

    return render_template('othello.html', board=current_board, current_player=current_player,move_action=move_action ,num_1=num_1,num_2=num_2)


@app.route('/model4', methods=['GET', 'POST'])
def game_model4():
    
    global current_player, current_board , model_4 # グローバル変数として現在のプレイヤーと盤面を使用

    num_2 = ' YOU'
    num_1 = ' CPU(France_先手モデル)'
    move_action = "/model4"

    if request.method == 'GET':
        make_cpu_move_black(current_board,model_4)


    if request.method == 'POST':
        current_board = np.array(current_board)
        if(request.form['row']=='' or request.form['col'] == ''):
            return render_template('othello.html', board=current_board, current_player=current_player,num_1=num_1,num_2=num_2)
        else:
            row = int(request.form['row'])
            col = int(request.form['col'])

        if current_player == 2:
            # ゲームのルールに従って石を置く処理（人間が手を打つ場合）
            if function.is_valid_move(current_board, row, col, 2):
                function.make_move(current_board, row, col, 2)
            else:
                warning = 'そこに石を置けません'
                return render_template('othello.html', board=current_board, current_player=current_player, warning=warning,num_1=num_1,num_2=num_2)

        # プレイヤーを切り替える
        current_player = 1 if current_player == 2 else 2
        
        # CPUの手を決定して石を置く処理（プレイヤーが'2'の場合）
        if current_player == 1:
            make_cpu_move_black(current_board,model_4)

    return render_template('othello.html', board=current_board, current_player=current_player,move_action=move_action,num_1=num_1,num_2=num_2)




if __name__ == '__main__':
    app.run()
