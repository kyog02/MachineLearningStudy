import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

X = np.random.randn(100,3)
t = np.random.randint(-100,100 ,size=(100,3)) * X

a = np.array([5,4,3])
print(np.square(np.sum(a))) # 
print(np.sum(a) ** 2)