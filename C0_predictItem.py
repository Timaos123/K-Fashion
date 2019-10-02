#coding:utf8

import pickle as pkl
from B0_training import *

if __name__=="__main__":

    with open("model/myModel"+time.strftime("%Y%m%d", time.localtime(time.time()))+".pkl","rb") as myModelFile:
        myModel=pkl.load(myModelFile)

    # myModel.plotModel()
    i=1
    priceListStr=""
    revenue=0
    while priceListStr!="exit":
        
        if i>1:
            priceListStr=input("enter price splitted by ',':")
            if priceListStr=="exit":
                break
            statusArr=np.array([int(numItem) for numItem in priceListStr.split(",")])
            tmpRevenue=price*np.sum(statusArr)
            print("本周收益：",tmpRevenue)
            revenue+=tmpRevenue
            myNode,price=myModel.predictItem(priceListStr,clayer=i-1,myNode=myNode)
            print("第{}周预测：{}".format(i,price))
        else:
            myNode,price=myModel.predictItem(priceListStr,clayer=0)
            print("第{}周预测：{}".format(i,price))
        i+=1

    print("总收益：",revenue)

    print("saving model ...")
    myModel.saveModel()

    print("plotting model ...")
    myModel.plotModel()

    print("finished !")