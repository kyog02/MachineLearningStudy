import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

# モデルをインポート
from LinearRegression import LinearRegression

# 部屋の広さから賃料を予測する。
# グラフの装飾
def plotData(X, t, y):
    plt.xlim([0,25]) #x軸の範囲を指定
    plt.ylim([0,35]) #y軸の範囲を指定 
    plt.xlabel('部屋の広さ[畳]',fontsize=14) #x軸のラベル
    plt.ylabel('家賃 [万円]', fontsize=14) #y軸のラベル
    plt.title('部屋の広さと賃料の関係', fontsize=16) #グラフのタイトル
    plt.scatter(X, t, c='b') #散布図
    plt.plot(X, y, c='black') #仮説
    plt.draw() #グラフを表示
    plt.show()


#　入力データ
# 部屋の間取り
X = np.array([4.5, 5, 4.5, 5.5, 5.8, 6, 7, 7, 7.2, 8, 8.2, 10, 11, 11.8, 12, 13, 13.5, 14, 14.2, 15, 15.2, 16, 17, 21, 23])
# 間取りに対する賃料
t = np.array([3, 4, 10, 9, 11, 7, 4.5, 10, 13, 12, 11, 17, 15, 22, 14, 21, 14, 15, 16, 28, 30, 18, 20, 22, 26])

# モデル生成
model = LinearRegression()

# 学習結果を記録する配列
cost = []

for i in range(10000):
    #仮説の計算
    y = model.predict(X)

    #目的関数（コスト関数）の値を記録
    J = model.cost_function(y,t)
    cost.append(J)
    
    #パラメータの更新(学習) 
    model.gradient_descent(X,y,t)

# 学習の経過
#plt.plot(cost[5000:10000])
#plt.show()

# 学習結果
#plotData(X, t, y)

#===========================================

# 実施
size = 1
price = model.predict(size)
print("部屋の広さ{}畳の賃料の相場は{:.2f}万円".format(size, price))
#===========================================

x = np.array([[1,2,3],[4,5,6]])
y = np.random.randn(10,3)
print(x)
print(x.shape)
print(x.size)
print(y)
print(y.shape)
print(y.size)