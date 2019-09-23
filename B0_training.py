#%%
import numpy as np
import tqdm
import pandas as pd
from copy import deepcopy
from pyecharts.charts import Tree
#%%
class Node:
    def __init__(self,myValue=0,ucb=np.inf,childList=[],parent=None):
        self.value=myValue
        self.checked=0
        self.ucb=ucb
        self.childList=childList.copy()
        self.parent=parent

class DecisionNode(Node):
    def __init__(self,decision,**kwargs):
        Node.__init__(self,**kwargs)
        self.decision=decision

class StatusNode(Node):
    def __init__(self,singleValue,status=[],**kwargs):
        Node.__init__(self,**kwargs)
        self.singleValue=singleValue
        self.status=status

#%%
class MCT(list):
    
    def __init__(self,maxLayer=12,maxNum=10,ensemble=False,C=2):
        self.maxLayer=maxLayer
        self.maxNum=maxNum
        self.ensemble=ensemble

        self.root=StatusNode(0,status=[0,0,0,maxNum])
        self.root.value=0
        self.nodeList=[]
        self.meanProfit=0
        self.stdProfit=0
        self.C=C

    def checkChild(self,checkedNode,checkedStatus):
        if len(checkedNode.childList)==0:
            return False
        if type(checkedNode)==int:
            myNode=self.nodeList[checkedNode]
        else:
            myNode=checkedNode
        for childI in checkedNode.childList:
            if self.nodeList[childI].status==checkedStatus:
                return True
        return False

    def findChild(self,checkedNode,checkedStatus):
        if len(checkedNode.childList)==0:
            return None
        if type(checkedNode)==int:
            myNode=self.nodeList[checkedNode]
        else:
            myNode=checkedNode
        for childI in checkedNode.childList:
            if self.nodeList[childI].status==checkedStatus:
                return self.nodeList[childI]
        return None

    def training(self,myX,myY,maxIter=5):
        '''
        myX:ValueB+ReturnB
        '''
        decisionList=list(set(myY.tolist()))
        print("training ...")
        self.nodeList=[self.root]
        
        #start iteration
        for iter in range(maxIter):
            #-for every iteration
            print("第{}轮迭代开始".format(iter+1))
            newStatusList=[self.root]
            for layerI in range(self.maxLayer):
                
                print("第{}周开始".format(layerI+1))

                #--for every layer
                
                #--new data
                newData=myX[iter*self.maxLayer+layerI,:]

                #--next week
                #--decision nodes
                newDecisionList=[]

                if self.maxNum>0:
                    #--the SKU was sold out
                    if len(newStatusList)==0:
                        break
                    #--whether to generalize the decision node
                    for statusNodeItem in newStatusList:
                        if len(statusNodeItem.childList)==0:
                            #-----if the status doesn't have decision child
                            for decisionItem in decisionList:
                                #---update checked of the status node
                                statusNodeItem.checked+=1

                                #------set the decision node
                                tmpDecisionNode=DecisionNode(decisionItem,parent=statusNodeItem)
                                tmpDecisionNode.checked+=1
                                #------add the node to the node list
                                self.nodeList.append(tmpDecisionNode)

                                #------add the node to the child list
                                statusNodeItem.childList.append(len(self.nodeList)-1)

                                #-----add the node to the newDecisionNodeList
                                newDecisionList.append(tmpDecisionNode)

                        else:
                            for decisionI in statusNodeItem.childList:
                                #---update checked of the status node
                                statusNodeItem.checked+=1
                                #-----add the node to the newDecisionNodeList
                                self.nodeList[decisionI].checked+=1
                                newDecisionList.append(self.nodeList[decisionI])
                    
                    #--set the newStatusList
                    newStatusList=[]
                    #--find the decision node with the largest ucb
                    maxUCB=max([newDecisionNodeItem.ucb for newDecisionNodeItem in newDecisionList])
                    #--status nodes
                    for decisionNodeItem in newDecisionList:
                        if decisionNodeItem.ucb==maxUCB:
                            #---status in this situation
                            if decisionNodeItem.parent.status[3]-np.sum(newData[:7]>=decisionNodeItem.decision)>=0:
                                #----if there is enough deposit
                                tmpStatus=[np.sum(newData[:7]>=decisionNodeItem.decision),\
                                            np.sum(newData[:7]<decisionNodeItem.decision),\
                                            np.sum(newData[7:]>0),\
                                            decisionNodeItem.parent.status[3]-np.sum(newData[:7]>=decisionNodeItem.decision)]
                            else:
                                #----if there is not enough deposit
                                tmpStatus=[decisionNodeItem.parent.status[3],\
                                            7-decisionNodeItem.parent.status[3],\
                                            np.sum(newData[7:]>0),\
                                            0]
                            if tmpStatus[0]+tmpStatus[1]!=7:
                                print(newData,decisionNodeItem.decision)
                            #---the decision doesn't have the status node
                            if self.checkChild(decisionNodeItem,tmpStatus)==False:

                                #----new status node
                                tmpStatusNode=StatusNode(decisionNodeItem.decision,\
                                                        status=tmpStatus,\
                                                        parent=decisionNodeItem)

                                #----add the status node into node list
                                self.nodeList.append(tmpStatusNode)

                                #----add the status node into the child list of the decision node
                                decisionNodeItem.childList.append(len(self.nodeList)-1)

                            else:
                                
                                #-----if the status already exists, change tmpNode into it
                                tmpStatusNode=self.findChild(decisionNodeItem,tmpStatus)
                            
                            #----update the node's value
                            tmpStatusNode.value=decisionNodeItem.decision*tmpStatusNode.status[0]

                            #----update the parents' value
                            tmpTravelNode=tmpStatusNode
                            tmpStatusNode.checked+=1
                            tmpValue=0
                            while tmpTravelNode.parent is not None:
                                if type(tmpTravelNode)==StatusNode:
                                    tmpValue=tmpTravelNode.value+tmpValue
                                else:
                                    tmpTravelNode.value=max(tmpTravelNode.value,tmpValue)
                                    tmpValue=tmpTravelNode.value
                                tmpTravelNode=tmpTravelNode.parent

                            #------add the node to newStatusList
                            if tmpStatusNode not in newStatusList and tmpStatusNode.status[3]>0:
                                newStatusList.append(tmpStatusNode)

                            #------update the ucb if find the tail node
                            if tmpStatusNode.status[3]==0:
                                #--backpropagation for this iteration
                                self.BPIter(tmpStatusNode)

                #only leave the status node with decision node with highest ucb
                maxUcb=0
                for statusNodeItem in newStatusList:
                    if statusNodeItem.parent.ucb>maxUcb:
                        maxUcb=statusNodeItem.parent.ucb
                newStatusI=0
                while newStatusI<len(newStatusList):
                    if newStatusList[newStatusI].parent.ucb<maxUcb:
                        newStatusList.pop(newStatusI)
                        newStatusI-=1
                    newStatusI+=1

                print("第{}周结束".format(layerI+1))

    def BPIter(self,myNode):
        tmpNode=myNode
        while tmpNode.parent is not None:
            
            #-update the ucb while BP
            tmpNode.ucb=tmpNode.value+self.C*np.sqrt(np.log(tmpNode.parent.checked)/tmpNode.checked)

            #-update the usb of the cousins of the tmpNode
            for childItem in tmpNode.parent.childList:
                if tmpNode is not self.nodeList[childItem]:
                    self.nodeList[childItem].ucb=self.nodeList[childItem].value+self.C*np.sqrt(np.log(self.nodeList[childItem].parent.checked)/self.nodeList[childItem].checked)

            tmpNode=tmpNode.parent

    def integrateTree(self,myNode):
        myDict={}
        if type(myNode)==DecisionNode:
            myDict["name"]="D"
        else:
            myDict["name"]="S status:{} price:{}".format(myNode.status,myNode.singleValue)
        myDict["children"]=[]
        if len(myNode.childList)!=0:
            for childI in myNode.childList:
                myDict["children"].append(self.integrateTree(self.nodeList[childI]))
        return myDict

    def plotModel(self):
        self.tree=Tree()
        self.echartTree=[self.integrateTree(self.root)]
        self.tree.add("", self.echartTree)
        self.tree.render()
#%%
itemName="A"
C=2

trainDf=pd.read_csv("data/Simulated_Data1.csv")
trainDf=trainDf.loc[trainDf["SKU"]==itemName,:]
X=np.array(trainDf.loc[:,[keyItem for keyItem in trainDf.keys() if keyItem.startswith("ValueB") or keyItem.startswith("ReturnB")]])
y=np.array(trainDf["price"])
myMCT=MCT()
myMCT.training(X,y,maxIter=5)
myMCT.plotModel()
#%%
