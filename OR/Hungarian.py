

from .Format import displayHelper
import numpy as np
from pandas import DataFrame
from itertools import combinations

class Hungarian():
    
    @classmethod
    def new(cls,table):
        df = DataFrame(table)
        df.columns = [i+1 for i in range(df.shape[0])]
        df.index = [i+1 for i in range(df.shape[0])]
        return Hungarian(df)
    
    def __init__(self,costDF,reducedDF=False,assignedDF=False):
        self.costDF = DataFrame(costDF).copy()

        if type(reducedDF) != bool:
            self.reducedDF = DataFrame(reducedDF).copy()
        else:
            self.reducedDF = (DataFrame(costDF).copy()
                            .apply(lambda x:x-min(x),axis=1)
                            .apply(lambda x:x-min(x),axis=0))

        if type(assignedDF) != bool:
            self.assignedDF = DataFrame(assignedDF).copy()
        else:
            self.assignedDF = False
        

    def solve(self,echo=False):
        if type(self.isOptimal()) != bool:
            return self.isOptimal()
        coveredDF = self.coverLines(echo=echo)
        return self.reduceDF(coveredDF,echo=echo).solve(echo=echo)
        

    def isOptimal(self):
        zeros = ()
        # locate zeros
        for colIndex in self.reducedDF.columns:
            for rowIndex in self.reducedDF.index:
                if self.reducedDF.loc[rowIndex,colIndex] == 0:
                    zeros += ((rowIndex,colIndex),)

        # check that there is a feasible combination
        for zero in combinations(zeros,len(self.reducedDF.columns)):
            x = set(i[0] for i in zero)
            y = set(i[1] for i in zero)
            if (len(x) == len(self.reducedDF.columns) 
            and len(y) == len(self.reducedDF.columns)):
                assignedDF = DataFrame(np.zeros_like(self.costDF))
                assignedDF.columns = self.costDF.columns
                assignedDF.index = self.costDF.columns
                for coord in zero:
                    assignedDF.loc[coord[0],coord[1]] = 1
                return Hungarian(self.costDF,self.reducedDF,assignedDF)
        return False

    def coverLines(self,echo=False):
        '''
        return a DataFrame where crossed out columns are represented by np.inf
        '''
        workingDF = self.reducedDF.copy()

        #axis{index (0), columns (1)}
        rows = (workingDF == 0).astype(int).sum(axis=1)
        cols = (workingDF == 0).astype(int).sum(axis=0)
        max0 = max(rows.append(cols))

        while max0 >0:
            if max0 in rows.array:
                for index, value in rows.items():
                    if value == max0:
                        workingDF.loc[index] = [np.inf] * workingDF.shape[1]
                        break
            else: # max0 in cols
                for index, value in cols.items():
                    if value == max0:
                        workingDF.loc[:,index] = [np.inf] * workingDF.shape[0]
                        break
            rows = (workingDF == 0).astype(int).sum(axis=1)
            cols = (workingDF == 0).astype(int).sum(axis=0)
            max0 = max(rows.append(cols))
        
        if echo:{displayHelper(workingDF)}
        return workingDF

    def reduceDF(self,coveredDF,echo=False):
        rows = (coveredDF == np.inf).astype(int).sum(axis=1)
        cols = (coveredDF == np.inf).astype(int).sum(axis=0)
        maskedRows = [index for index, value in rows.items() if value == coveredDF.shape[1]]
        maskedCols = [index for index, value in cols.items() if value == coveredDF.shape[1]]
        minUncovered = coveredDF.min().min()

        workingDF = self.reducedDF.copy()
        for colIndex in workingDF.columns:
            for rowIndex in workingDF.index:
                if colIndex in maskedCols and rowIndex in maskedRows:
                    workingDF.loc[rowIndex,colIndex] += minUncovered
                elif colIndex not in maskedCols and rowIndex not in maskedRows:
                    workingDF.loc[rowIndex,colIndex] -= minUncovered
        if echo:{displayHelper(workingDF)}
        return Hungarian(self.costDF,workingDF)
        
    def objectiveValue(self):
        '''
        return calculated objective value of this transport tableau using current assignments
        '''
        if type(self.assignedDF) == bool:
            return self.costDF.mul(self.assignDF).sum().sum()
        else:
            raise ValueError("assignments have not been calculated")

    def __repr__(self):
        if type(self.assignedDF) == bool:
            return f"Hungarian\n{str(self.costDF)}\n"
        else:
            return f"Assignments\n{str(self.assignedDF)}\n"