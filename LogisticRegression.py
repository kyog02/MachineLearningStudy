import numpy as np
import random

class LogisticRegression:
    def __init__(self,learning_rate=0.01):
        self.W = random.uniform(0,5) # 0以上5未満の値を一様分布で生成
        self.b = random.uniform(0,5) # 0以上5未満の値を一様分布で生成
        self.learning_rate = learning_rate

    # 仮説関数(ロジスティック回帰はシグモイド関数)
    # 注意！：ロジスティック回帰の仮設関数ことシグモイド関数と
    #        ニューラルネットワークの活性化関数のシグモイド関数は微妙に計算式が違う
    #        ⇒NNの方は、活性化関数前に重みとバイアスの行列積とってるから。やってることは同じ。
    # 計算式：1/1 + exp(-(x * W + b)) ※exp部分は(-ΘTx)と書くらしい。。
    def predict(self,x):
        z = x * self.W + self.b
        h = 1 / (1 + np.exp(-z))
        return h

    # 目的(コスト)関数(ロジスティック回帰は尤度関数と呼ばれる？)
    def cost_function(self,y,t):
        m = t.shape[0]
        J = -1 / m * np.sum( t * np.log(y) + (1 - t) * np.log(1 - y))
        return J

    # パラメータ更新関数(最急降下法)
    # パラメータの更新は線形回帰と同じ
    def gradient_descent(self,X,y,t):
        m = y.size
        W = self.W - self.learning_rate / m * np.sum((y - t) * X)
        self.W = W
        b = self.b - self.learning_rate / m * np.sum(y - t)
        self.b = b

# 多項式用のロジスティック回帰
class LogisticRegressionMulti:
    def __init__(self,learning_rate=0.01):
        self.W = None
        self.b = None
        self.learning_rate = learning_rate

    # 仮説関数(ロジスティック回帰はシグモイド関数)
    # 多項式はxの
    def predict(self,X, theta):
        m = X.shape[0] #訓練例の数
        XwithBias = np.c_[np.ones([m,1]),X]         # Xの1列目にズラッと1を並べる XwithBias.shapeは(100, 3)
        y = 1/ (1 + np.exp(-XwithBias.dot(theta)))  # シグモイド関数
        return y

    # 目的(コスト)関数(ロジスティック回帰は尤度関数と呼ばれる？)
    def cost_function(self,theta, X, t):
        y = predict(X, theta) #仮説h
        m = X.shape[0] #訓練例の数
        # 1次元のnumpy配列同士のdot積は内積を表します。
        J = - (t.dot(np.log(y))+(1- t).dot(np.log(1- y))) / m
        # -inf * 0 は nan になる。そのときはJ全体をinfで返す。
        if np.isnan(J):
            return np.inf
        return J

    #正則化された目的関数
    def costReg(theta, X, t, lmd):
        m = t.size
        J = cost_function(self,theta, X, t)
        J = J + lmd * np.square(np.linalg.norm(theta[1:])) / (2 * m)
        return J

    # パラメータ更新関数(最急降下法)
    # パラメータの更新は線形回帰と同じ
    def gradient_descent(self,theta,X,t):
        m = X.shape[0] #訓練例の数
        XwithBias = np.c_[np.ones([m,1]),X] # Xの1列目にズラッと1を並べる XwithBias.shapeは(100, 3)
        y = predict(X, theta) # 仮設関数
        # ヒント：XwithBiasを使います。この行列の転置はXwithBias.T
        grad =  XwithBias.T.dot(y - t) / m 
        self.grad = grad
        return grad

    # パラメータ更新関数 + L2正則化
    def gradReg(theta, X, t, lmd):
        m = t.size
        g = gradient_descent(theta,X, t)
        theta[0] = 0 #0番目の成分は正則化項に含めないため、0にしておきます。
        g = g + lmd * theta[0] / (2 * m)
        return g