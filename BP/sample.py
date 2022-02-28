from NeuronNetwork import NeuronLayer
from NeuronNetwork import NeuronNet
from NeuronNetwork import Sigmoid

def sample():

    #学习异或
    X = [[0,0,-1],[0,1,-1],[1,0,-1],[1,1,-1]]
    Y=[0,1,1,0]
    inn = 2
    innerlayers = []
    print("正在准备输入层第一层和隐藏层第一层:")
    innerlayers.append(NeuronLayer(2,inn,0.2,Sigmoid,lambda s:s*(1-s)))
    #增加max(i)层隐含层
    for i in range(1):
        print("正在准备输入层第一层和隐藏层第%d层"%(i))
        innerlayers.append(NeuronLayer(inn,inn,0.1))
    outputlayer = NeuronLayer(inn,1,0.1,Sigmoid,lambda s:s*(1-s))
    outputlayer.ChangeOutput()

    print("创建全连接层神经网络(BP)")
    nnet = NeuronNet(innerlayers,outputlayer)

    times = 30000
    for i in range(times):
        if i%100 == 0 or i == times-1:
            print("train",i,"times")
            #测试
            me = 0
            testres = []
            for x,y in zip(X,Y):
                outres = nnet.Active(x)
                testres.append(outres[0][0])
                me += abs(outres[0][0] - y)
            me = me / len(X)
            print(me)
            if me < 0.05:
                print(testres)
                break
        #训练
        nnet.Train(X,Y)

if __name__ == "__main__":
    sample()






