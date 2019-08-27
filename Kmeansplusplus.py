import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
# K-means
# 未実装だが、k-means++について
# K-meansは重心の位置をランダムに設定するが、k-means++は違う
# 初期のk個のクラスタ中心はなるべく離れている方が良いという考えにもとづいている。
# まず始めにデータ点をランダムに選び1つ目のクラスタ中心とし、全てのデータ点とその最近傍のクラスタ中心の距離を求め、
# その距離の二乗に比例した確率でクラスタ中心として選ばれていないデータ点をクラスタ中心としてランダムに選ぶ。
X = np.random.randn(300,2)


def plotData(X, idx=np.ones(X.shape[0],dtype=float)):
    map = plt.get_cmap("rainbow")
    idxn = idx.astype("float") / max(idx.astype("float"))
    colors = map(idxn)
    plt.scatter(X[:, 0], X[:, 1], 15, marker="o", c=colors, edgecolors=colors)
    plt.draw()

def plotProgress(X, centroids, idx=np.ones(X.shape[0],dtype=float)):
    #plt.hold(True)
    plotData(X, idx)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x",s=200,linewidths=4, c="g")
    plt.show()

#====================================================================================

# 中心とそのほかすべてのデータ点の距離を算出
def distance(a,b):
    return np.sum(np.square(b-a),axis=1)

# 最初は重心(centroid)をランダムに決める
# 注意：ランダム初期化によって、収束する箇所が決まるため、50 - 1000回程度ランダム初期化して、最も多く同じ収束箇所にとどまったパターンを採用する。
#      これはクラスタ数(K)が少ない場合に有効。クラスタ数が何百となる場合は1回でほぼ大域的最小値付近に落ちてくれる。
# X:入力データ
# K:重心の数(クラスタを何個に分けるか)
def InitCentroids(X, K):
    m, n = X.shape
    centroids_idx = []
    idx = np.random.randint(m)
    centroids_idx.append(idx)
    closest_distance = distance(X[idx],X)
    l = np.sum(closest_distance)
    for i in range(1,k):
        # ルーレット選択を用いて確率的に新しいサンプルを取得
        rand_val = np.random.random_sample() * l  # numpy.random.random_sample(): 0.0以上、1.0未満
        idx = np.searchsorted(np.cumsum(closest_distance),rand_val)
        centroids_idx.append(idx)
        # 近傍距離をアップデート
        new_distance = distance(X[idx],X)
        closest_distance = np.maximum(closest_distance,new_distance)
        l = np.sum(closest_distance)

    centroids = X[centroids_idx]   
    return centroids

# 目的関数に近いイメージ
# 各訓練例に対して最も近い重心を探す
def findClosestCentroids(X, centroids):
    K,_ = centroids.shape
    m, n = X.shape
    idx = np.empty(m, dtype=int) # 入力データの列数(300)の分だけ、空配列用意
    for i in range(m):
        norm = np.empty(centroids.shape[0])
        for j in range(K):
            norm[j] = np.linalg.norm(X[i,:] - centroids[j,:]) # X[i] - centroids[j]のL2ノルム
        idx[i] = np.argmin(norm)
    return idx

# K平均法
# 教師あり学習でいうパラメータの更新のイメージ
# 次に重心を移すべき点を計算
# 注意：もしある重心が何れの訓練集合からも遠いところに置かれるなどして、どの訓練例もその重心のクラスタに所属しなかった場合、
# 　　　次に重心を移すべき点を計算することができません。この場合、もう一度ランダム初期化を行い一からやり直すか、
# 　　　そのクラスタをなくし重心の個数を減らす必要があります。
# 　　　後者のケースは、課題３の最後の行~np.isnan(clusterMeans).any(axis=1)によって実現されています。
def computeClusterMeans(X, idx):
    m, n = X.shape
    K = idx.max()+1
    clusterMeans = np.array([X[idx==i,:].mean(axis=0) for i in range(K)])
    return clusterMeans[~np.isnan(clusterMeans).any(axis=1)] #nanを含む行は除外して返す


k = 3
centroids = InitCentroids(X, k) # 第二引数は重心(クラスタ)の数
idx = findClosestCentroids(X, centroids)
plotProgress(X,centroids,idx)
for i in range(1,6):
    centroids = computeClusterMeans(X, idx)
    idx = findClosestCentroids(X, centroids)
    #plotProgress(X,centroids,idx)

