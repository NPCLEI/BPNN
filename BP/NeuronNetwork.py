import matplotlib.pyplot as plt
import numpy as np
import math
import random
import cv2
import sys      
import os   
from tqdm import tqdm



def ClearConsole():
    f_handler=open('out.log', 'w')      # 打开out.log文件
    oldstdout = sys.stdout              # 保存默认的Python标准输出
    sys.stdout=f_handler                # 将Python标准输出指向out.log
    os.system('cls')                    # 清空Python控制台       
    sys.stdout = oldstdout              # 恢复Python默认的标准输出

def Draw(inputdata,func):
    for i in inputdata:
        plt.plot([i[0]-0.01,i[0]+0.01],[i[1]+0.01,i[1]-0.01])
        plt.plot([i[0]-0.01,i[0]+0.01],[i[1]-0.01,i[1]+0.01])

    last = [-15,func(-15)]
    for j in range(-150,230):
        i = [j/10,func(j/10)]
        plt.plot([i[0],last[0]],[i[1],last[1]])
        last = i        

    plt.show()

def ReadData(filepath):
    f = open(filepath,"rb")
    x1 = f.readline().decode('utf8').split(',')
    x2 = f.readline().decode('utf8').split(',')
    dataset = []
    for i,j in zip(x1,x2):
        dataset.append([float(i),float(j),-1])
    return dataset


#激活函数
def Sigmoid(value):
    try:
        return 1/(1+math.pow(math.e,-1*value))
    except:
        if -1*value > 700:
            return 0
        else:
            return 1

def ReLu(value):
    if value < 0:
        return 0
    else:
        return value

#激活函数
def ActiveFunc(array:np.array,func = Sigmoid):
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j]=func(array[i][j])
    return array

#矩阵函数
def FunArray(array:np.array,func):
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j]=func(array[i][j])
    return array

#激活函数的导数
def SigmoidDerivative(value):
    return Sigmoid(value)*(1-Sigmoid(value))

#产生[-1，1]的随机数
def TherRandom():
    return random.randint(-100,100)/100.0

class NeuronLayer:
    def __init__(self,dim_input:int,num_neurons:int,alp:float,activeFunc=ReLu,afd=lambda s:np.where(s>0,1,0.01)):
        """
        :param:dim_input:该层输入向量维度
        :num_neurons:该层神经元数量
        :alp:梯度下降步长
        :activeFunc:激活函数(参数是一个值)
        :afd:激活函数倒数(参数是输出向量)
        """
        weightMatx = np.full((1,dim_input+1),TherRandom())
        print("准备中.......")
        for i in tqdm(range(num_neurons-1)):
            t = np.full((1,dim_input+1),TherRandom())
            weightMatx = np.row_stack((weightMatx,t[-1]))

        self.num_neurons = num_neurons
        self.weightMatx = weightMatx
        self.alp = alp
        self.category = "normal"
        self.activeFunc = activeFunc
        self.afd = afd

    def ChangeOutput(self):
        '''
            change this layer to output layer
            将该层转换为输出层
        '''
        self.category = 1

    def PrintWeightMatx(self):
        for i in self.weightMatx:
            print(i)

    def SetInput(self,input:list):
        self.input=input

    def SetEorr(self,eorr:list):
        """
        设置偏差
        """
        self.eorr = eorr 

    def CalcuEorrMatix(self):
        t = self.weightMatx.transpose()
        #return t*self.eorr
        if len(t[0])<len(self.eorr):
            self.eorr=self.eorr[:-1]
        return np.dot(t,self.eorr)

    def Output(self):
        result = ActiveFunc(np.dot(self.weightMatx,self.input),self.activeFunc)
        if self.category == "normal":
            res = np.append(result,[[-1]],axis=0)
        else:
            res = result
        self.outputres = res
        return res

    def Adjust(self):
        s = self.outputres
        # ds = self.alp*s*(1-s)
        ds = self.alp * self.afd(s) * s
        c = ds*self.eorr
        if len(c)>len(self.weightMatx):
            c=c[:-1]
        wempoint = 0
        tw = []
        j=0
        for i in c:
           tw.append((i[0]*(self.input.transpose()))[0])
           j+=1
        self.weightMatx=self.weightMatx+np.array(tw)

class NeuronNet:
    def __init__(self,innerlayers,outputlayer):
        self.innerlayers = innerlayers
        self.outputlayer = outputlayer
        self.datapoint=0

    def Train(self,data,Y):
        """
            反向传播
        """
        innerlayers = self.innerlayers
        outputlayer = self.outputlayer
        for ii,yi in zip(data,Y):
            #print("训练数据：%d"%(self.datapoint))
            #self.datapoint+=1
            i=np.array([ii])
            i=i.transpose()
            for j in range(len(innerlayers)):
                innerlayers[j].SetInput(i)
                i=innerlayers[j].Output()
            outputlayer.SetInput(i)
            outmatx = outputlayer.Output()

            # eorr = abs(outmatx**0.5 - np.array(yi)**0.5)
            eorr = np.array(yi) - outmatx

            # print("eorr",eorr)

            outputlayer.SetEorr(eorr)
            outputlayer.Adjust()

            innerlayers[-1].SetEorr(outputlayer.CalcuEorrMatix())
            for j in range(len(innerlayers)-2,-1,-1):
                innerlayers[j].SetEorr(innerlayers[j+1].CalcuEorrMatix())
                innerlayers[j].Adjust()

    def Active(self,input)->list:
        """
        前向传播
        return 输出结果
        """
        innerlayers = self.innerlayers
        outputlayer = self.outputlayer
        i=np.array([input])
        i=i.transpose()
        for j in range(len(innerlayers)):
            innerlayers[j].SetInput(i)
            i=innerlayers[j].Output()
        outputlayer.SetInput(i)
        return outputlayer.Output()