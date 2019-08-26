import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
# PCA 主成分分析

def plotX(X):
    plt.scatter(X[:,0], X[:,1], marker="o")
    plt.axis([0.5, 6.5, 2, 8])
    plt.axis("equal")
    plt.draw()

# 入力値
X = np.random.randn(50,2)

# 各特長のレンジを揃える
# 返値：Feature Scaling後のデータであるX_normと、元の各特徴の平均mu、標準偏差std（元の値に戻すときに利用）
def featureScaling(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = ( X - mu ) / std
    return [X_norm, mu, std]

# 特異値分解を行う関数
# 返値は Σ=USVΣ=USV の U と S
# 共分散行列 Σ は半正定値行列ですから、実は固有値分解 ΣΣ=VWV**−1 の結果と等価です。
# （実装によって符号が反転する場合はある）
def runPCA(X):
    m, n = X.shape
    Sigma = X.T.dot(X) / m #共分散行列
    U, S, V = np.linalg.svd(Sigma)
    return [U, S]

# 固有値が大きかった固有ベクトル（主成分）に対して、射影していく様子をプロットしましょう。
# これによって、特徴の数を2つから1つに落とせるわけです。
# もとの特徴に復元するときは、他の成分を落としている限り完全には復元されず、次に挙げる例のちょうど射影した場所に復元されます。

# 射影
def projectData(X, U, K):
    return np.array([[ X[i,:].dot(U[:,k]) for k in range(K)] for i in range(X.shape[0])])

# 復元
def recoverData(Z, U, K):
    return np.array([[Z[i,:].dot(U[j,:K]) for j in range(U.shape[0])] for i in range(Z.shape[0])])

X_norm, mu, std = featureScaling(X)
U, S = runPCA(X_norm)


# 得たものをプロットしましょう。データの平均付近から順番に各固有ベクトルを掃引します。
# 長さは固有値に比例するようプロットする。
tmp0 = mu + (2 * S[0] * U[:,0].T)*std
tmp1 = mu + (2 * S[1] * U[:,1].T)/std

plotX(X)
plt.plot([mu[0], tmp0[0]], [mu[1], tmp0[1]], linewidth=4, c="r")
plt.plot([mu[0], tmp1[0]], [mu[1], tmp1[1]], linewidth=4, c="r")
plt.show()


K = 1 #1次元に落としたい
Z = projectData(X_norm, U, K)

X_rec = recoverData(Z, U, K)

plt.scatter(X_norm[:, 0], X_norm[:, 1], marker='o', color="b") #簡便のため、すでにFeature Scalingされたデータで見ることにします。
plt.axis([-4, 3, -4, 3])
plt.axis('equal')
plt.scatter(X_rec[:,0], X_rec[:,1], marker="o", color="r")
for i in range(len(X_norm)):
    plt.plot([X_norm[i,0], X_rec[i,0]], [X_norm[i,1], X_rec[i,1]])
plt.show()