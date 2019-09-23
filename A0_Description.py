#%%
#Loading data
import pandas as pd
import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#load simulation data
trainDf=pd.read_csv("data/Simulated Data.csv")

#select seasons
seasonList=set(trainDf["Season"])
# print(seasonList)

# #transform original data into one-dimensional array
# ValueList=[keyItem for keyItem in trainDf.keys() if keyItem.startswith("ValueB")==True]

# #get the list of Value data of every season
# seasonDataList=[np.array(trainDf.loc[(trainDf["Season"]==seasonItem) & (trainDf["SKU"]=="C"),ValueList]).reshape(1,-1)[0] for seasonItem in seasonList]


# #%%
# #get the p of homogeneity of variance test of every season
# varPMat=np.matrix([[stats.levene(seasonDataItem1,seasonDataItem2)[1] for seasonDataItem2 in seasonDataList] for seasonDataItem1 in seasonDataList])
# varPMat[varPMat>0.05]==1
# varPMat[varPMat!=1]==0
# plt.matshow(varPMat)
# plt.colorbar()
# plt.show()

# #%%
# #get the p of T-test of every season
# tPMat=np.matrix([[stats.ttest_ind(seasonDataList[seasonDataI1],seasonDataList[seasonDataI2],equal_var=varPMat[seasonDataI1,seasonDataI2]==1)[1] for seasonDataI2 in range(len(seasonDataList))] for seasonDataI1 in range(len(seasonDataList))])
# tPMat[tPMat>0.1]==1
# tPMat[tPMat!=1]==0
# plt.matshow(tPMat)
# plt.colorbar()
# plt.show()

#%%
itemName="A"
trainDf=trainDf.loc[(trainDf["Season"]==1)&(trainDf["SKU"]==itemName),[keyItem for keyItem in trainDf.keys() if keyItem.startswith("ValueB")]]
# print(trainDf)
pltList=[]
for keyItem in trainDf.keys():
    pltList.append(\
        plt.plot([i+1 for i in range(np.array(trainDf[keyItem]).shape[0])],np.array(trainDf[keyItem]))[0]\
    )
plt.legend(pltList,[str(i) for i in range(1,8)])
plt.show()


#%%
