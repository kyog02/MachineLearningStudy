import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

from LogisticRegression import LogisticRegression

def plotData(X,y):
    ytick = np.array(['正常 ０', '異常 １'])#目盛りのラベルの名前を指定（文字列）
    plt.yticks([0,1] ,ytick, fontsize=14)#y軸の目盛りにラベルを表示
    plt.xlabel('ある製品試験に不合格だった個数[個] ',fontsize=14) #x軸のラベル
    plt.title('製品試験と工場（正常・異常）の関係', fontsize=16) #グラフのタイトル
    plt.scatter(X[:11] ,y[:11], c='b') #散布図
    plt.scatter(X[11:] ,y[11:], c='r') #散布図
    plt.show() #グラフを表示


# 分類問題のデータセットXを作成
m = 22 # 不合格だった個数の最大数
X = np.arange(0, m, 1) # 0からm-1個の不合格だった個数を生成

# 分類問題のデータセット（正解ラベル）を作成
t = np.where(X > 10, 1, 0)

model = LogisticRegression()
iteration = 10000
cost = []

for i in range(iteration):
    # 仮説の計算
    y = model.predict(X)
    # 目的関数の計算
    J = model.cost_function(y,t)
    cost.append(J)
    # パラメータの更新
    model.gradient_descent(X,y,t)

# 学習の経過をグラフ化
plt.plot(cost, c='b')
plt.xlabel('学習回数')
plt.ylabel('目的関数の出力の値')
plt.show()

# 学習結果を
ytick = np.array(['正常 ０', '異常 １'])#目盛りのラベルの名前を指定（文字列）
plt.yticks([0, 1], ytick, fontsize=14)
plt.ylim([-0.1, 1.1]) # y軸の幅
plt.xlabel('ある製品試験に不合格だった個数[個] ',fontsize=14) #x軸のラベル
plt.title('製品試験と工場（正常・異常）の関係', fontsize=16) #グラフのタイトル
plt.scatter(X[:11], t[:11], c='b') #散布図
plt.scatter(X[11:], t[11:], c='r') #散布図
plt.plot(X, y, c='black', label='仮説（シグモイド関数）') #仮説（シグモイド関数）
plt.legend(loc='upper left')
plt.show()


# 学習が済んだ状態で線形回帰の仮説で出力すると、決定境界がわかる。
# y2 = X *model.W + model.b
# ytick = np.array(['正常 ０', '異常 １'])#目盛りのラベルの名前を指定（文字列）
# plt.yticks([0, 1], ytick, fontsize=14)
# plt.ylim([-0.1, 1.1]) # y軸の幅
# plt.xlabel('ある製品試験に不合格だった個数[個] ',fontsize=14) #x軸のラベル
# plt.title('製品試験と工場（正常・異常）の関係', fontsize=16) #グラフのタイトル
# plt.scatter(X[:11], t[:11], c='b') #散布図
# plt.scatter(X[11:], t[11:], c='r') #散布図
# plt.plot(X, y2, c='black', label='仮説（シグモイド関数）') #仮説（シグモイド関数）
# plt.legend(loc='upper left')
# plt.show()