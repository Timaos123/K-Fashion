#coding:utf8
import time
import pandas as pd
import numpy as np
from B0_training import *
if __name__=="__main__":

    batchsize=5

    time1=time.time()
    iter=1
    minLoss=np.inf
    cr=[i for i in range(500,2000,500)]
    ci=0

    #load data ...
    trainDf=pd.read_csv("data/Simulated_Data1.csv")
    
    #get keyList ...
    keyList1=[keyItem for keyItem in trainDf.keys() if keyItem.startswith("ValueB")]
    keyList2=[keyItem for keyItem in trainDf.keys() if keyItem.startswith("ReturnB")]
    keyList=keyList1+keyList2
    X=np.array(trainDf.loc[:,keyList])
    y=np.array(trainDf["price"])

    with open("data/evaluation_record.txt","w+",encoding="utf8") as evaluationFile:
        while ci<len(cr):

            #abstract data ...
            anchorList=np.random.randint(0,50,batchsize).tolist()
            trainX=np.zeros((batchsize*12,X.shape[1]))
            i=0
            for anchor in anchorList:
                trainX[i*12:i*12+12,:]=X[anchor*12:anchor*12+12,:]
                i+=1

            trainY=np.zeros(batchsize*12)
            i=0
            for anchor in anchorList:
                trainY[i*12:i*12+12]=y[anchor*12:anchor*12+12]
                i+=1

            #training
            myMCT=MCT(C=cr[ci])
            myMCT.training(trainX,trainY,maxIter=batchsize)
            rndIter=np.random.randint(0,50)

            #testing
            newDataList=X[rndIter*12:rndIter*12+12,:].tolist()
            yList=y[rndIter*12:rndIter*12+12]
            newDataList=[[]]+newDataList
            # print(newDataList,yList)

            #evaluating
            i=0
            priceListStr=""
            totalDicisionLoss=0
            for row in newDataList:
                print("第{}周预测".format(i))
                if i>0:
                    row=np.array(row)
                    myStatus=[np.sum(row[row>=yPre][:7]),\
                                np.sum(row[row>=yPre][7:14]),\
                                np.sum(row[row>=yPre][14:21])]
                    myNode,yPre=myMCT.predictItem(myStatus,i,myNode=myNode)

                    decisionLoss=np.sum(row[:21]>=y[i])*y[i]-yPre*np.sum(row[:21]>=yPre)
                    totalDicisionLoss+=decisionLoss
                    print("迭代次数：{},当前c:{},预测价：{},真实价格：{},最高利润：{},实际利润：{}\n".format(iter,cr[ci],y[ci],yPre,np.sum(row[:21]>=y[i])*y[i],yPre*np.sum(row[:21]>=yPre)))
                else:
                    myNode,yPre=myMCT.predictItem(row,i)
                i+=1
            print("saving ...")
            evaluationFile.write("迭代次数：{},当前c:{},预测价：{},真实价格：{},决策损失：{}\n".format(iter,cr[ci],y[ci],yPre,decisionLoss))
            if totalDicisionLoss<minLoss:
                ci-=1
                myMCT.saveModel()
            ci+=1
            iter+=1
    print(time2-time1)