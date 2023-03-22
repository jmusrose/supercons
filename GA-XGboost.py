# -*-coding:utf-8 -*-
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

class GA_XGBoost(object):
    
    def __init__(self,population_size,chromosome_length,max_value,pc,pm):
        self.population_size=population_size
        self.choromosome_length=chromosome_length
        self.max_value=max_value
        self.pc=pc
        self.pm=pm
        self.times = 0
        df =pd.read_csv('CleanData.csv')
        x = df.drop('critical_temp', axis = 1)
        sc = StandardScaler()
        x_scale = sc.fit_transform(x)
        x = pd.DataFrame(x_scale, columns = x.columns)
        y = df['critical_temp']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
    
    
    #初始化染色体
    def species_origin(self):
        population = [[]]
        for i in range(self.population_size):
            temporary = []

            for j in range(sum(self.choromosome_length)):
                temporary.append(random.randint(0,1))
            
            population.append(temporary)
        return population[1:]
    
    #二进制转为十进制
    def translation(self, population):
        temporarys = [[]]
        for i in range(len(population)):
            t = 0
            temporary = []
            for j in range(len(self.choromosome_length)):
                total = 0
                for k in range(self.choromosome_length[j]):
                    total += population[i][t] * (math.pow(2, k))
                    t = t + 1
                temporary.append(total)
            temporarys.append(temporary)
        return temporarys[1:]


    def funcs(self, para):
        reg = XGBR(n_estimators = int(para[0] + 1)
        , max_depth = int(para[1] + 1)
        , eta = para[2]
        # , subsample = 0.7 + para[3]
        ).fit(self.x_train, self.y_train)
        y_prediction = pd.Series(reg.predict(self.x_test))
        min_rmse = round(np.sqrt(mean_squared_error(self.y_test, y_prediction)), 4)
        r2 = round(r2_score(y_prediction, self.y_test), 4)
        self.times = self.times + 1
        with open('1.txt', 'a') as f:
            strs = "颗数:" + str(para[0]) + "\t最大深度:" + str(para[1]) + "\tlr:" + str(para[2]) + "\t" + str(r2) + "\t" + str(min_rmse) + "\t" + str(self.times) + "\n"
            f.write(strs)
        # print("颗数:",'%.3f'%para[0], "   最大深度:", '%.3f'%para[1], "  lr:",'%.3f'%para[2],"  sub:",para[3], "    ", r2,  "   ", min_rmse)
        if(r2 <= 0):
            return 0.1
        return r2

#   利用目标函数进行筛选
    def function(self,population):
        function1=[]
        temporary=self.translation(population)
        # print(temporary)
        for i in range(len(temporary)):
            ans_list = []
            for j in range(len(temporary[0])):
                x = temporary[i][j] * self.max_value[j]/(math.pow(2,self.choromosome_length[j])-1)
                ans_list.append(x)
            function1.append(self.funcs(ans_list))
        return function1

    #计算适应度斐伯纳且列表
    def cumsum(self,fitness1):
        for i in range(len(fitness1)-2,-1,-1):
        # range(start,stop,[step])
        # 倒计数
            total=0
            j=0
 
            while(j<=i):
                 total+=fitness1[j]
                 j+=1
 
            fitness1[i]=total
            fitness1[len(fitness1)-1]=1

    #计算适应度和
 
    def sum(self,fitness_value):
        total=0
 
        for i in range(len(fitness_value)):
            total+=fitness_value[i]
        return total

    def best(self,population,fitness_value):
 
        px=len(population)
        bestindividual=population[0]
        bestfitness=fitness_value[0]
        # print(fitness_value)
        # print(population)
        for i in range(1,px):
   # 循环找出最大的适应度，适应度最大的也就是最好的个体
            if(fitness_value[i]>bestfitness):
 
               bestfitness=fitness_value[i]
               bestindividual=population[i]
 
        return [bestindividual,bestfitness]

    def selection(self,population,fitness_value):
        new_fitness=[]
    #单个公式暂存器
        total_fitness=self.sum(fitness_value)
    #将所有的适应度求和
        for i in range(len(fitness_value)):
            new_fitness.append(fitness_value[i]/total_fitness)
    #将所有个体的适应度正则化
        self.cumsum(new_fitness)
        ms=[]
    #存活的种群
        population_length=pop_len=len(population)
    #求出种群长度
    #根据随机数确定哪几个能存活
 
        for i in range(pop_len):
            ms.append(random.random())      
    # 产生种群个数的随机值
        ms.sort()
    # 存活的种群排序
        fitin=0
        newin=0
        new_pop=population
        lists = []
 
    #轮盘赌方式
        while newin<pop_len:
                # print(ms[newin], "      ", new_fitness[fitin])
                if(ms[newin]<new_fitness[fitin]):
                    new_pop[newin]=population[fitin]
                    newin+=1
                else:
                    fitin+=1
        population=new_pop      

    def crossover(self,population):
#pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体，这里没用
        pop_len=len(population)
        for i in range(pop_len-1):
            
            if(random.random()<self.pc):
 
                cpoint=random.randint(0,len(population[0]))
           #在种群个数内随机生成单点交叉点
                temporary1=[]
                temporary2=[]
                # print(i)
                # print(self.b2d(population[i]), '          ', self.b2d(population[i + 1]))
                # print(population[i], '          ', population[i + 1])
                temporary1.extend(population[i][0:cpoint])
                temporary1.extend(population[i+1][cpoint:len(population[i])])
           #将tmporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因，
           #然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
 
                temporary2.extend(population[i+1][0:cpoint])
                temporary2.extend(population[i][cpoint:len(population[i])])
        # 将tmporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因，
        # 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
                population[i]=temporary1
                population[i+1]=temporary2
                # print(self.b2d(population[i]), '          ', self.b2d(population[i + 1]))
                # print(population[i], '          ', population[i + 1])
                
        # 第i个染色体和第i+1个染色体基因重组/交叉完成

    # 将每一个染色体都转化成十进制 max_value,再筛去过大的值
    def b2d(self,best_individual):
        temporarys = []
        t = 0
        for i in range(len(self.choromosome_length)):
            total = 0
            for j in range(self.choromosome_length[i]):
                total += best_individual[t] * (math.pow(2, j))
                t = t + 1
            total=total*self.max_value[i]/(math.pow(2,self.choromosome_length[i])-1)
            temporarys.append(total)
        return temporarys




    #定义适应度
    def fitness(self,function1):
 
        fitness_value=[]
 
        num=len(function1)
 
        for i in range(num):
            # print('fitness1:', function1[i])
            if(function1[i]>0):
                temporary=function1[i]
            else:
                temporary=1
            # print('fitness2:', function1[i])
        # 如果适应度小于0,则定为0
 
            fitness_value.append(temporary)
        #将适应度添加到列表中
 
        return fitness_value

    def mutation(self,population):
     # pm是概率阈值
         px=len(population)
    # 求出种群中所有种群/个体的个数
         py=len(population[0])
    # 染色体/个体基因的个数
         for i in range(px):
             if(random.random()<self.pm):
                mpoint=random.randint(0,py-1)
            #
                if(population[i][mpoint]==1):
               #将mpoint个基因进行单点随机变异，变为0或者1
                   population[i][mpoint]=0
                else:
                   population[i][mpoint]=1

    def plot(self, results):
        X = []
        Y = []
 
        for i in range(20):
            X.append(i)
            Y.append(results[i][0])
 
        plt.plot(X, Y)
        plt.show()
    
    def main(self):
 
        results = [[]]
        fitness_value = []
        fitmean = []
 
        population = self.species_origin()
 
        for i in range(100):
            function_value = self.function(population)
            # print('fit funtion_value:',function_value)
            fitness_value = self.fitness(function_value)
            # print('fitness_value:',fitness_value)
 
            best_individual, best_fitness = self.best(population,fitness_value)
            results.append([best_fitness, self.b2d(best_individual)])
        # 将最好的个体和最好的适应度保存，并将最好的个体转成十进制,适应度函数
            self.selection(population,fitness_value)
            self.crossover(population)
            self.mutation(population)
            print('舒适度',best_fitness)
        results = results[1:]
        # print(results)
        results.sort()
        self.plot(results)


    

population_size=10
max_value=[1000, 10, 0.2, 0.3]
chromosome_length=[10, 4, 7, 7]
pc=0.6
pm=0.2
ga = GA_XGBoost(population_size,chromosome_length,max_value,pc,pm)
ga.main()
