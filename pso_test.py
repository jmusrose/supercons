import random
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class parti(object):
    def __init__(self, v, x):
        self.v = v                    
        self.x = x                    
        self.pbest = x                

class PSO(object):
    def __init__(self, interval, part_num, tab = 'max', w = 1, c1 = 2, c2 = 2, partisNum=10, iterMax=20):
        self.interval = interval
        self.tab = tab.strip()
        self.iterMax = iterMax
        self.w = w
        self.c1, self.c2 = c1, c2
        self.v_max = [(i[1] - i[0]) * 0.1 for i in interval]
        self.partisNum = partisNum
        self.part_Num = part_num
        sc = StandardScaler()
        df = pd.read_csv('CleanData.csv')
        x = df.drop('critical_temp', axis = 1)
        x_scale = sc.fit_transform(x)
        x = pd.DataFrame(x_scale, columns = x.columns)
        y = df['critical_temp']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
        #-------------------------------------------------------
        self.partis_list, self.gbest = self.initPartis()

    def initPartis(self):
        partis_list = []
        gbest = []
        for i in range(self.partisNum):
            v_seeds = [random.uniform(-self.v_max[j], self.v_max[j]) for j in range(self.part_Num)]
            x_seeds = [random.uniform(*self.interval[j]) for j in range(self.part_Num)]
            partis_list.append(parti(v_seeds, x_seeds))
        temp = 'find_' + self.tab
        if hasattr(self, temp):
                gbest = getattr(self, temp)(partis_list)
        else:
            exit('>>>tab标签传参有误："min"|"max"<<<')
        return partis_list, gbest

    def func(self, x):
        reg = XGBR(n_estimators = int(x[0] + 1),
                   max_depth = int(x[1] + 1),
                   eta = x[2]
                    ).fit(self.x_train, self.y_train)
        y_prediction = pd.Series(reg.predict(self.x_test))
        min_rmse = round(np.sqrt(mean_squared_error(self.y_test, y_prediction)), 4)
        r2 = round(r2_score(y_prediction, self.y_test), 4)
        with open('pso.txt', 'a') as f:
            f.write(f'{min_rmse}\t{r2}\n')
        return min_rmse

    def update(self, partis):
        gbest = self.gbest
        for p in partis:
            r1, r2 = np.random.random(2)
            p.v = self.w * p.v + self.c1 * r1 * (p.pbest - p.x) + self.c2 * r2 * (gbest - p.x)
            p.x += p.v
            p.x = self.clip(p.x)
            temp = self.func(p.x)
            if temp < self.func(p.pbest):
                p.pbest = p.x
                if temp < self.func(gbest):
                    gbest = p.x
        return gbest

    def clip(self, x):
        for i in range(self.part_Num):
            if x[i] > self.interval[i][1]:
                x[i] = self.interval[i][1]
            elif x[i] < self.interval[i][0]:
                x[i] = self.interval[i][0]
        return x

    def find_max(self, partis):
        m = -1e9
        gbest = []
        for p in partis:
            temp = self.func(p.x)
            if temp > m:
                m = temp
                gbest = p.x
        return gbest

    def find_min(self, partis):
        m = 1e9
        gbest = []
        for p in partis:
            temp = self.func(p.x)
            if temp < m:
                m = temp
                gbest = p.x
        return gbest

    def run(self):
        for i in range(self.iterMax):
            self.gbest = self.update(self.partis_list)
        return self.gbest

pso = PSO(interval = [(10, 1000), (1, 10), (0.01, 0.99)], part_num = 3, tab = 'min', w = 0.9, c1 = 2, c2 = 2)
pso.run()

