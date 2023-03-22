import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from time import time
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 定义“粒子”类
class parti(object):
    def __init__(self, v, x):
        self.v = v                    # 粒子当前速度
        self.x = x                    # 粒子当前位置
        self.pbest = x                # 粒子历史最优位置

class PSO(object):
    def __init__(self, interval, part_num, tab = 'max', w = 1, c1 = 2, c2 = 2, partisNum=10, iterMax=30):
        self.interval = interval
        self.tab = tab.strip()
        self.iterMax = iterMax
        self.w = w
        self.c1, self.c2 = c1, c2
        self.v_max = self.get_vmax(interval)
        self.partisNum = partisNum
        self.part_Num = part_num
        self.times = 0
        sc = StandardScaler()
        df = pd.read_csv('CleanData.csv')
        x = df.drop('critical_temp', axis = 1)
        x_scale = sc.fit_transform(x)
        x = pd.DataFrame(x_scale, columns = x.columns)
        y = df['critical_temp']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
        #-------------------------------------------------------
        self.partis_list, self.gbest = self.initPartis()

    def get_vmax(self, interval):
        temp = []
        for i in interval:
            x = (i[1] - i[0]) * 0.1
            temp.append(x)
        return temp
    
    def initPartis(self):
        partis_list = [[]]
        gbest = []
        for i in range(self.partisNum):
            v_seeds = []
            x_seeds = []
            for j in range(self.part_Num):
                v_seed = random.uniform(-self.v_max[j], self.v_max[j])
                x_seed = random.uniform(*self.interval[j]) 
                v_seeds.append(v_seed)
                x_seeds.append(x_seed)
            partis_list.append(parti(v_seeds, x_seeds))
        partis_list = partis_list[1:]
        temp = 'find_' + self.tab
        if hasattr(self, temp):
                gbest = getattr(self, temp)(partis_list)
        else:
            exit('>>>tab标签传参有误："min"|"max"<<<')
        return partis_list[1:], gbest

    def func(self, x):
        reg = XGBR(n_estimators = int(x[0] + 1),
                   max_depth = int(x[1] + 1),
                   eta = x[2]
                    ).fit(self.x_train, self.y_train)
        y_prediction = pd.Series(reg.predict(self.x_test))
        min_rmse = round(np.sqrt(mean_squared_error(self.y_test, y_prediction)), 4)
        r2 = round(r2_score(y_prediction, self.y_test), 4)
        self.times = self.times + 1
        with open('pso.txt', 'a') as f:
            strs = "颗数:" + str(x[0]) + "\t最大深度:" + str(x[1]) + "\tlr:" + str(x[2]) + "\t" + str(r2) + "\t" + str(min_rmse) + "\t" + str(self.times) + "\n"
            f.write(strs)
        print(x, "   $R^2$:", r2, "    RMSE:", min_rmse)
        return r2

    # def find_max(self, partis_list):
    #     temp = []
    #     for i in range(self.part_Num):
    #         x = -1000
    #         for j in range(len(partis_list)):
    #             x = max(x, partis_list[j].x[i])
    #         temp.append(x)
    #     return temp
    #chatGTP优化代码
    def find_max(self, partis_list):
        return [max(particle.x[i] for particle in partis_list) for i in range(self.part_Num)]


    # def solve(self):
    #     for i in range(self.iterMax):
    #         for j in range(self.part_Num):
    #             for parti_c in self.partis_list:
    #                 f1 = self.func(parti_c.x)
    #                 parti_c.v[j] = self.w * parti_c.v[j] + self.c1 * random.random() * (parti_c.pbest[j] - parti_c.x[j]) + self.c2 * random.random() * (self.gbest[j] - parti_c.x[j])
    #                 if parti_c.v[j] > self.v_max[j]:
    #                     parti_c.v[j] = self.v_max[j]
    #                 elif parti_c.v[j] < -self.v_max[j]:
    #                     parti_c.v[j] = -self.v_max[j]
    #                 if self.interval[j][0] <= parti_c.x[j] + parti_c.v[j] <= self.interval[j][1]:
    #                     parti_c.x[j] = parti_c.x[j] + parti_c.v[j]
    #                 else:
    #                     parti_c.x[j] = parti_c.x[j] - parti_c.v[j]
    #                 f2 = self.func(parti_c.x)
    #                 getattr(self, 'deal_'+self.tab)(f1, f2, parti_c)
    ####chatGTP优化的代码
    def solve(self):
        for i in range(self.iterMax):
            for j in range(self.part_Num):
                for k in range(len(self.partis_list)):
                    parti_c = self.partis_list[k]
                    f1 = self.func(parti_c.x)
                    parti_c.v[j] = self.w * parti_c.v[j] + self.c1 * random.random() * (parti_c.pbest[j] - parti_c.x[j]) + self.c2 * random.random() * (self.gbest[j] - parti_c.x[j])
                    parti_c.v[j] = max(-self.v_max[j], min(parti_c.v[j], self.v_max[j]))
                    if self.interval[j][0] <= parti_c.x[j] + parti_c.v[j] <= self.interval[j][1]:
                        parti_c.x[j] = parti_c.x[j] + parti_c.v[j]
                    else:
                        parti_c.x[j] = parti_c.x[j] - parti_c.v[j]
                    f2 = self.func(parti_c.x)
                    getattr(self, 'deal_'+self.tab)(f1, f2, parti_c)


    def deal_max(self, f1, f2, parti):
        if f2 > f1:
            parti.pbest = parti.x
        if f2 > self.func(self.gbest):
            self.gbest = parti.x

interval = [[1, 1000],[1, 10], [0.01, 1]]
Psos = PSO(interval, 3)
partis_list, gbest = Psos.initPartis()
Psos.solve()