#coding:utf8

import pandas as pd
import numpy as np

def setPrice(myDf,decisionList):
    neededArr=np.array(myDf.loc[:,[keyItem for keyItem in myDf.keys() if keyItem.startswith("ValueB")]])
    resultList=np.matrix([np.sum((neededArr>=decisionItem)*neededArr*decisionItem,axis=1) for decisionItem in decisionList]).T.tolist()
    return [decisionList[row.index(max(row))] for row in resultList]

def trans2Nan(x):
    if x==0:
        return np.nan
    else:
        return x

def fillNA(x,myMean,myStd):
    if np.isnan(x)==True:
        return myMean+myStd*(np.random.rand()*2-1)
    else:
        return x

if __name__=="__main__":

    print("loading data ...")
    myDf=pd.read_csv("data/Simulated Data.csv",encoding="utf8")

    print("inserting y ...")
    decisionList=[99,199,299,399,499,599,699,799,899,999]
    myDf["price"]=setPrice(myDf,decisionList)

    print("filling missing data ...")
    keyList=[keyItem for keyItem in myDf.keys() if keyItem.startswith("ValueB")]
    for keyItem in keyList:
        myDf[keyItem]=myDf[keyItem].apply(lambda x:trans2Nan(x))
    
    myDf["mean"]=myDf.loc[:,keyList].mean(axis=1)
    myDf["std"]=myDf.loc[:,keyList].std(axis=1)
    for keyItem in keyList:
        myDf[keyItem]=myDf.apply(lambda x:fillNA(x[keyItem],x["mean"],x["std"]), axis=1)

    print("saving data ...")
    myDf.to_csv("data/Simulated_Data1.csv",index=None)

    print("finished !")