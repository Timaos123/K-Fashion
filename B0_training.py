#%%
import numpy as np
import tqdm
import pandas as pd
from copy import deepcopy
from pyecharts.charts import Tree
import pickle as pkl
import time
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
        self.status=status#[buyers,leavers,returners,rest_of_SKU]

#%%
class MCT():
    
    def __init__(self,maxLayer=12,maxNum=10,ensemble=False,C=2):
        self.maxLayer=maxLayer
        self.maxNum=maxNum
        self.ensemble=ensemble

        self.root=StatusNode(0,status=[0,0,0,0,0,0,maxNum,maxNum,maxNum])
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
        if type(checkedStatus)==np.ndarray:
            checkedStatus=checkedStatus.tolist()
        if len(checkedNode.childList)==0 or checkedNode is None:
            return None
        if type(checkedNode)==int:
            myNode=self.nodeList[checkedNode]
        else:
            myNode=checkedNode
        for childI in checkedNode.childList:
            if self.nodeList[childI].status[:3]==checkedStatus:
                return self.nodeList[childI]
        return None

    def updateChecked(self,myNode:DecisionNode):
        tmpTravelNode=myNode
        while tmpTravelNode is not None:
            tmpTravelNode.checked+=1
            tmpTravelNode=tmpTravelNode.parent

    def BPIter(self,myNode,C=500):
        tmpNode=myNode
        while tmpNode.parent is not None:
            
            #-update the ucb while BP
            tmpNode.ucb=tmpNode.value+self.C*np.sqrt(np.log(tmpNode.parent.checked)/tmpNode.checked)

            #-update the usb of the cousins of the tmpNode
            for childItem in tmpNode.parent.childList:
                if tmpNode is not self.nodeList[childItem]:
                    self.nodeList[childItem].ucb=self.nodeList[childItem].value+self.C*np.sqrt(np.log(self.nodeList[childItem].parent.checked)/self.nodeList[childItem].checked)

            tmpNode=tmpNode.parent

    def findReturn(self,myDecisionNode,statusI):
        tmpNode=myDecisionNode
        returnList=[]
        while tmpNode.parent!=None:
            if type(tmpNode)==StatusNode:
                if tmpNode.status[statusI]!=0:
                    returnList.append(tmpNode.status[statusI])
            tmpNode=tmpNode.parent
        returnArr=np.array(returnList)
        return returnArr

    def training(self,myX,myY,startLayer=0,maxIter=5,incrementalNode=None):
        '''
        myX:ValueB+ReturnB
        '''
        decisionList=list(set(myY.tolist()))
        print("training ...")
        if incrementalNode==None:
            self.nodeList=[self.root]
            startNode=self.root
        else:
            startNode=incrementalNode

        newDecisionList=[]
        
        #start iteration
        for iter in tqdm.tqdm(range(maxIter)):

            #-for every iteration
            print("第{}轮迭代开始".format(iter+1))
            if np.sum(startNode.status[-3:])==0:
                break
            newStatusList=[startNode]
            for layerI in range(startLayer,self.maxLayer):
                
                # print("第{}周开始".format(layerI+1))

                #--for every layer
                
                #--new data
                newData=myX[iter*self.maxLayer+layerI,:]

                #--next week
                #--decision nodes
                newDecisionList=[]

                if self.maxNum>0:
                    #--the SKU was sold out
                    if len(newStatusList)==0:
                        print("the SKU was sold out in all the path")
                        break
                    #--whether to generalize the decision node
                    for statusNodeItem in newStatusList:
                        if len(statusNodeItem.childList)==0:
                            #-----if the status doesn't have decision child
                            for decisionItem in decisionList:

                                #------set the decision node
                                tmpDecisionNode=DecisionNode(decisionItem,parent=statusNodeItem)

                                #------update the checked on the path
                                self.updateChecked(tmpDecisionNode)

                                #------add the node to the node list
                                self.nodeList.append(tmpDecisionNode)

                                #------add the node to the child list
                                statusNodeItem.childList.append(len(self.nodeList)-1)

                                #-----add the node to the newDecisionNodeList
                                newDecisionList.append(tmpDecisionNode)

                        else:
                            for decisionI in statusNodeItem.childList:
                                #---update checked of the node
                                tmpTravelNode=statusNodeItem
                                self.updateChecked(self.nodeList[decisionI])
                                newDecisionList.append(self.nodeList[decisionI])
                    
                    #--set the newStatusList
                    newStatusList=[]
                    #--find the decision node with the largest ucb
                    maxUCB=max([newDecisionNodeItem.ucb for newDecisionNodeItem in newDecisionList])
                    #--status nodes
                    for decisionNodeItem in newDecisionList:
                        if decisionNodeItem.ucb==maxUCB:
                            #---status in this situation
                            #A,B,C,Ar,Br,Cr,Al,Bl,Cl
                            if layerI==11:
                                #----update return status in the last week
                                newDataApp=[self.findReturn(decisionNodeItem,i) for i in range(3,6)]
                                tmpStatus=[min(np.sum(np.append(newData[:7],newDataApp[0])>=decisionNodeItem.decision),decisionNodeItem.parent.status[6]),\
                                            min(np.sum(np.append(newData[7:14],newDataApp[1])>=decisionNodeItem.decision),decisionNodeItem.parent.status[7]),\
                                            min(np.sum(np.append(newData[14:21],newDataApp[2])>=decisionNodeItem.decision),decisionNodeItem.parent.status[8]),\
                                            0,\
                                            0,\
                                            0,\
                                            max(decisionNodeItem.parent.status[6]-np.sum(newData[:7]>=decisionNodeItem.decision),0),\
                                            max(decisionNodeItem.parent.status[7]-np.sum(newData[7:14]>=decisionNodeItem.decision),0),\
                                            max(decisionNodeItem.parent.status[8]-np.sum(newData[14:21]>=decisionNodeItem.decision),0)]
                            else:
                                tmpStatus=[min(np.sum(newData[:7]>=decisionNodeItem.decision),decisionNodeItem.parent.status[6]),\
                                            min(np.sum(newData[7:14]>=decisionNodeItem.decision),decisionNodeItem.parent.status[7]),\
                                            min(np.sum(newData[14:21]>=decisionNodeItem.decision),decisionNodeItem.parent.status[8]),\
                                            int(np.sum(np.sum((newData[21:28]>0)*(newData[:7]>=decisionNodeItem.decision))/7*newData[21:28])),\
                                            int(np.sum(np.sum((newData[28:35]>0)*(newData[7:14]>=decisionNodeItem.decision))/7*newData[28:35])),\
                                            int(np.sum(np.sum((newData[35:42]>0)*np.sum(newData[14:21]>=decisionNodeItem.decision))/7*newData[35:42])),\
                                            max(decisionNodeItem.parent.status[6]-np.sum(newData[:7]>=decisionNodeItem.decision),0),\
                                            max(decisionNodeItem.parent.status[7]-np.sum(newData[7:14]>=decisionNodeItem.decision),0),\
                                            max(decisionNodeItem.parent.status[8]-np.sum(newData[14:21]>=decisionNodeItem.decision),0)]
                                        
                            #---the decision doesn't have the status node
                            tmpStatusNode=self.findChild(decisionNodeItem,tmpStatus)
                            if tmpStatusNode==None:

                                #----new status node
                                tmpStatusNode=StatusNode(decisionNodeItem.decision,\
                                                        status=tmpStatus,\
                                                        parent=decisionNodeItem)

                                #----add the status node into node list
                                self.nodeList.append(tmpStatusNode)

                                #----add the status node into the child list of the decision node
                                decisionNodeItem.childList.append(len(self.nodeList)-1)

                            #----update the parents' value
                            self.updateChecked(tmpStatusNode)
                            tmpTravelNode=tmpStatusNode
                            tmpAggValue=0
                            while tmpTravelNode.parent is not None:
                                if type(tmpTravelNode)==StatusNode:
                                    #-----status node
                                    tmpTravelNode.value=(tmpTravelNode.value*tmpTravelNode.checked+decisionNodeItem.decision*tmpTravelNode.status[0]+decisionNodeItem.decision*tmpTravelNode.status[1]+decisionNodeItem.decision*tmpTravelNode.status[2]+tmpAggValue)/(tmpTravelNode.checked+1)
                                    tmpAggValue=tmpTravelNode.value
                                else:
                                    #-----decision node
                                    tmpTravelNode.value=(tmpTravelNode.value*tmpTravelNode.checked+np.mean([self.nodeList[childI].value*self.nodeList[childI].checked/tmpTravelNode.checked for childI in tmpTravelNode.childList]))/(tmpTravelNode.checked+1)
                                    # print(self.nodeList[tmpTravelNode.childList[0]].checked/tmpTravelNode.checked)
                                tmpTravelNode=tmpTravelNode.parent
                            #------add the node to newStatusList
                            if tmpStatusNode not in newStatusList\
                                     and (tmpStatusNode.status[6]>0\
                                     or tmpStatusNode.status[7]>0\
                                     or tmpStatusNode.status[8]>0):
                                newStatusList.append(tmpStatusNode)
                            elif tmpStatusNode.status[6]==0\
                                    and tmpStatusNode.status[7]==0\
                                    and tmpStatusNode.status[8]==0:
                                    #------update the ucb if find the tail node
                                self.BPIter(tmpStatusNode,C=self.C)

                    #only keep the status node with decision node with highest ucb
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
        
                # print("第{}周结束".format(layerI+1))
            # #------update the ucb if find the tail node
            # for tmpStatusNode in newStatusList:
            #     if (tmpStatusNode.status[6]==0\
            #         and tmpStatusNode.status[7]==0\
            #         and tmpStatusNode.status[8]==0):
            #         #--backpropagation for this iteration
            #         self.BPIter(tmpStatusNode)
            #update the C

    def integrateTree(self,myNode):
        myDict={}
        if type(myNode)==DecisionNode:
            myDict["name"]="D,ucb:{:.2f},price:{},value:{:.2f},checked:{}".format(myNode.ucb,myNode.decision,myNode.value,myNode.checked)
        else:
            if myNode.parent!=None:
                myDict["name"]="S,status:{},value:{:.2f},weight:{:.2f}".format(myNode.status,myNode.value,myNode.checked/myNode.parent.checked)
            else:
                myDict["name"]="root"
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

    def saveModel(self):
        with open("model/myModel"+time.strftime("%Y%m%d", time.localtime(time.time()))+".pkl","wb+") as myModelFile:
            pkl.dump(self,myModelFile)

    def predictItem(self,statusListStr,clayer,myNode=None):
        if len(statusListStr)==0:
            #-no status, return the decision node with largest ucb in the first layer
            myNode=self.nodeList[0]
            ucbList=[self.nodeList[childI].ucb for childI in self.nodeList[0].childList]
            maxUCB=max(ucbList)
            for childI in self.nodeList[0].childList:
                if self.nodeList[childI].ucb==maxUCB:
                    return self.nodeList[childI],self.nodeList[childI].decision
        else:
            #-transform the statusListStr into a list
            if type(statusListStr)==str:
                statusList=[int(statusItem) for statusItem in statusListStr.split(",")]
            else:
                statusList=statusListStr
            
            #-find the status node
            myStatus=np.array(statusList)[:3]
            statusNode=self.findChild(myNode,myStatus)

            if statusNode is not None:
                if len(statusNode.childList)==0:
                    #---the status is tail
                    print("卑微预测（真的还要卖吗？你不为祖国母亲庆生的吗？）：",statusNode.parent.decision)
                    return statusNode.parent,statusNode.parent.decision
                #--find the status node, give the new decision
                maxUCB=max([self.nodeList[childI].ucb for childI in statusNode.childList])
                for childI in statusNode.childList:
                    if self.nodeList[childI].ucb==maxUCB:
                        decision=self.nodeList[childI].decision
                        print("预测：",decision)
                        return self.nodeList[childI],decision
            else:
                
                #--find the status node,regeneralize dataset
                trainDf=pd.read_csv("data/Simulated_Data1.csv")
                keyList1=[keyItem for keyItem in trainDf.keys() if keyItem.startswith("ValueB")]
                keyList2=[keyItem for keyItem in trainDf.keys() if keyItem.startswith("ReturnB")]
                keyList=keyList1+keyList2

                #--find the nearest status
                minDis=np.inf
                for xi1 in range(50):
                    if np.sum(abs(np.array(trainDf.loc[:,keyList])[xi1*12+clayer,:3]-myStatus))<minDis:
                        minDis=np.sum(abs(np.array(trainDf.loc[:,keyList])[xi1*12+clayer,:3]-myStatus))
                for xi1 in range(50):
                    if np.sum(abs(np.array(trainDf.loc[:,keyList])[xi1*12+clayer,:3]-myStatus))==minDis:
                        X1=np.array(trainDf.loc[:,keyList])[xi1*12:xi1*12+12]
                        y1=np.array(trainDf["price"])[xi1*12:xi1*12+12]
                        break

                #--a random status
                xi2=np.random.randint(0,high=50)

                X=np.concatenate((X1,\
                                np.array(trainDf.loc[:,keyList])[xi2*12:xi2*12+12]),\
                                axis=0)
                y=np.concatenate((y1,\
                                np.array(trainDf["price"])[xi2*12:xi2*12+12]),\
                                axis=0)

                #--new status node
                restA=myNode.parent.status[6]-myStatus[0]
                restB=myNode.parent.status[7]-myStatus[1]
                restC=myNode.parent.status[8]-myStatus[2]
                tmpStatus=[myStatus[0],myStatus[1],myStatus[2],0,0,0,restA,restB,restC]
                statusNode=StatusNode(myNode.decision,\
                                        status=tmpStatus,\
                                        parent=myNode)
                self.nodeList.append(statusNode)
                myNode.childList.append(len(self.nodeList)-1)
                statusIndex=len(self.nodeList)-1
                #--expand the tree
                self.training(X,y,startLayer=clayer+1,incrementalNode=statusNode,maxIter=2)
                statusNode=self.nodeList[statusIndex]

                if len(statusNode.childList)==0:
                    print("卑微预测（真的还要卖吗？你不为祖国母亲庆生的吗？）：",statusNode.parent.decision)
                    return statusNode.parent,statusNode.parent.decision
                
                for childI in statusNode.childList:
                    #---find reasonal virtual decision node
                    tmpDecisionNode=self.nodeList[childI]
                    if np.sum(statusNode.status[:3])>0:
                        #----status > 0
                        maxUCB=max([self.nodeList[childI].ucb for childI in statusNode.childList if self.nodeList[childI].decision<=statusNode.parent.decision])
                        if tmpDecisionNode.decision<=statusNode.parent.decision:
                            if tmpDecisionNode.ucb==maxUCB:
                                decision=tmpDecisionNode.decision
                                print("卑微预测：",decision)
                                return tmpDecisionNode,decision
                    else:
                        #----status==[0,0,0]
                        try:
                            maxUCB=max([self.nodeList[childI].ucb for childI in statusNode.childList if self.nodeList[childI].decision<statusNode.parent.decision])
                            if tmpDecisionNode.decision<statusNode.parent.decision:
                                if tmpDecisionNode.ucb==maxUCB:
                                    decision=tmpDecisionNode.decision
                                    print("卑微预测：",decision)
                                    return tmpDecisionNode,decision
                        except ValueError:
                            tmpDecisionNode=DecisionNode(99,parent=statusNode)
                            tmpDecisionNode.checked=1
                            self.nodeList.append(tmpDecisionNode)
                            statusNode.childList.append(len(self.nodeList)-1)
                            print("卑微预测： 99")
                            return tmpDecisionNode,99

#%%
if __name__=="__main__":
    C=2000

    trainDf=pd.read_csv("data/Simulated_Data1.csv")
    keyList1=[keyItem for keyItem in trainDf.keys() if keyItem.startswith("ValueB")]
    keyList2=[keyItem for keyItem in trainDf.keys() if keyItem.startswith("ReturnB")]
    keyList=keyList1+keyList2
    X=np.array(trainDf.loc[:,keyList])
    y=np.array(trainDf["price"])
    myMCT=MCT(C=C)
    myMCT.training(X,y,maxIter=1)

    print("saving model ...")
    myMCT.saveModel()

    print("plotting model ...")
    myMCT.plotModel()