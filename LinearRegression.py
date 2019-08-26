import numpy as np
import random

# 1変数の線形回帰
class LinearRegression:
    def __init__(self,learning_rate=0.01):
        self.W = random.uniform(0,5)
        self.b = random.uniform(0,5)
        self.learning_rate = learning_rate

    # 線形回帰の推測関数
    def predict(self,X):
        y = X * self.W + self.b
        return y

    # 線形回帰の目的関数)
    # 最小二乗法で実装(least_squares)
    def cost_function(self,y,t):
        m = t.size
        J = 1 / (2 * m) * np.sum((y - t) ** 2) 
        return J


    # パラメータ更新関数(最急降下法)
    def gradient_descent(self,X,y,t):
        m = y.size
        W = self.W - self.learning_rate / m * np.sum((y - t) * X)
        self.W = W
        b = self.b - self.learning_rate / m * np.sum(y - t)
        self.b = b



# L2正則化＋多変数の線形回帰の目的関数
def linearRegCostFunction(theta, XwithBias, y, lmd):
    m = y.size
    J = ((XwithBias.dot(theta) - y)**2).sum() / (2*m) + (lmd / (2*m)) * (theta[1:]**2).sum()
    return J

# L2正則化＋多変数の線形回帰の勾配
def linearRegGrad(theta, XwithBias, y, lmd):
    m = y.size
    grad = XwithBias.T.dot(XwithBias.dot(theta) - y) / m + (lmd / m) * np.r_[0, theta[1:]]
    return grad




# 多項式で特徴を増やす関数
def polyFeatures(X, p):
    m, = X.shape # Xは1-dim。mの後のカンマ必須
    X_poly = np.empty((m, p))

    for i in range(p):
        X_poly[:,i] = X**(i+1)

    return X_poly

# FeatureNormalize
# ググっても出てこないからでんと思う。。
def featureNormalize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = ( X - mu ) / sigma

    return [X_norm, mu, sigma]