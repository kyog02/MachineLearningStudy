import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

# モデルをインポート
from LinearRegression import LinearRegression


X = np.random.randn(100,1)
t = np.random.randint(-100,100 ,size=(100,1)) * X


model = LinearRegression()
cost = []

for i in range(1000):
    #仮説の計算
    y = model.predict(X)

    #目的関数（コスト関数）の値を記録
    J = model.cost_function(y,t)
    cost.append(J)
    
    #パラメータの更新(学習) 
    model.gradient_descent(X,y,t)

# plt.plot(cost)
# plt.show()

#===========================================
# 実施
peoples = 1
price = model.predict(peoples)
print("人口が+100人の時、予測される税収の増減は %d"% price, "百万円")
#===========================================


plt.xlim([-3,3])
plt.ylim([-300,300])
plt.scatter(X, t)
plt.scatter(X, y, color='orange')
plt.xlabel('街の人口の増減 [百人]')
plt.ylabel('税収の増減 [百万円]')
plt.show()

