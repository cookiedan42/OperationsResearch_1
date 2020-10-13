from .Format import *
import numpy as np
from pandas import DataFrame,Series


class TransportTable():
    @classmethod
    def new(cls,costs,colRemain,rowRemain,colName = "Remain",rowName = "Remain"):
        costDF = DataFrame(costs).copy()
        assignDF = DataFrame(np.zeros_like(costDF)).copy()
        for df in [costDF,assignDF]:
            df.columns = [i+1 for i in range(len(df.columns))]
            df.index = [i+1 for i in range(len(df.index))]
        
        colRemain = Series(colRemain,name=colName).copy()
        colRemain.index = colRemain.index = [i+1 for i in range(len(colRemain.index))]
        rowRemain = Series(rowRemain,name=rowName).copy()
        rowRemain.index = rowRemain.index = [i+1 for i in range(len(rowRemain.index))]
        return TransportTable(costDF,assignDF,rowRemain,colRemain)

    def __init__(self,costDF,assignDF,colRemain,rowRemain):
        self.costDF = costDF.copy()
        self.assignDF = assignDF.copy()
        self.colVals = colRemain.copy()
        self.rowVals = rowRemain.copy()

    def assign(self,rowNo,colNo,amount):
        if (amount<= self.colVals.loc[colNo]
        and amount<= self.rowVals.loc[rowNo]):

            newRowVals = self.rowVals.copy()
            newRowVals.loc[rowNo] -= amount

            newColVals = self.colVals.copy()
            newColVals.loc[colNo] -= amount

            newAssignDF = self.assignDF.copy()
            newAssignDF.loc[rowNo,colNo] += amount

            return TransportTable(self.costDF.copy(),newAssignDF,newColVals, newRowVals)
        else:
            raise ValueError("too large of an assignment") 
    
    def objectiveValue(self):
        return self.costDF.mul(self.assignDF).sum().sum()

    def northWest(self,echo=False):
        if echo == True:
            self.display()

        if self.rowVals.sum() + self.colVals.sum() ==0:
            #all assignment done
            return self
        
        for i in self.rowVals:
            if i!=0:
                rowIndex = 1+ self.rowVals.to_list().index(i)
                break
        for i in self.colVals:
            if i!=0:
                colIndex = 1+self.colVals.to_list().index(i)
                break
        amount = min(self.rowVals.loc[rowIndex],self.colVals.loc[colIndex])
        return self.assign(rowIndex,colIndex,amount).northWest(echo=echo)

    def minimumCost(self,echo=False,prevCost=0):
        if echo == True:
            self.display()
        if self.rowVals.sum() + self.colVals.sum() ==0:
            #all assignment done
            return self
        
        costs =  np.sort(np.unique(self.costDF.to_numpy().flatten()))
        filter_arr = costs > prevCost-1
        costs = costs[filter_arr]
        for cost in costs:
            for i in self.costDF.index:
                for j in self.costDF.columns:
                    if( self.costDF.loc[i,j] == cost
                        and self.rowVals.loc[i] != 0
                        and self.colVals.loc[j] != 0
                    ):
                        amount = min(self.rowVals.loc[i],self.colVals.loc[j])
                        return self.assign(i,j,amount).minimumCost(echo=echo,prevCost=cost)

    def vogel(self,echo=False):
        if echo:
            self.display()
        if self.rowVals.sum() + self.colVals.sum() ==0:
            #all assignment done
            return self

        rowOpp = []
        for i in range(len(self.rowVals)):
            rowArr = np.sort(np.unique(self.costDF.iloc[i,:]))
            if self.rowVals.iloc[i]==0 or len(rowArr)==1:
                rowOpp += [0]
            else:
                rowOpp +=[rowArr[1] - rowArr[0]]
        colOpp = []
        for i in range(len(self.colVals)):
            colArr = np.sort(np.unique(self.costDF.iloc[:,i]))
            if self.colVals.iloc[i]==0 or len(colArr)==1:
                colOpp += [0]
            else:
                colOpp +=[colArr[1] - colArr[0]]
        maxCost = max(rowOpp + colOpp)

        if maxCost in rowOpp:
            rowIndex = rowOpp.index(maxCost)+1
            toSearch = self.costDF.loc[rowIndex].copy()
            for i in self.colVals.index:
                if self.colVals.loc[i] == 0:
                    toSearch.loc[i] = np.inf
            for i in self.colVals.index:
                if toSearch.loc[i] == min(toSearch):
                    amount = min(self.rowVals.loc[rowIndex],self.colVals.loc[i])
                    return self.assign(rowIndex,i,amount).vogel(echo=echo)
        else: #maxCost in colOpp
            colIndex = colOpp.index(maxCost)+1
            toSearch = self.costDF.loc[:,colIndex].copy()
            for i in self.rowVals.index:
                if self.rowVals.loc[i] == 0:
                    toSearch.loc[i] = np.inf
            for i in self.rowVals.index:
                if toSearch.loc[i] == min(toSearch):
                    amount = min(self.rowVals.loc[i],self.colVals.loc[colIndex])
                    return self.assign(i,colIndex,amount).vogel(echo=echo)
    #display
    def displayFormat(self):
        dispDF =self.assignDF.copy()
        dispDF.insert(self.assignDF.shape[1],self.rowVals.name,self.rowVals)
        dispColVars = self.colVals.copy()
        dispColVars = dispColVars.append(Series([""],index=[self.rowVals.name]))
        dispColVars.name = self.colVals.name
        dispDF = dispDF.append(dispColVars)
        return dispDF
    def display(self):
        displayHelper(self.displayFormat())
    def __repr__(self):
        return str(self.displayFormat())