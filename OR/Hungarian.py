from .Format import display2
from .LP import LP
from .Constraint import Constraint
from .ObjectiveFunction import ObjectiveFunction
from .NonNeg import NonNeg

import numpy as np
from pandas import DataFrame,Series
from itertools import combinations

from typing import TypeVar,List,Tuple,Dict
Hungarian = TypeVar("Hungarian")

class Hungarian():
    
    @classmethod
    def new(cls,inArray:List[List[float]]) -> Hungarian:
        '''
        convert list of lists to DataFrame for Hungarian
        '''
        #data validity
        if not all([len(i)-1 == len(inArray[-1]) for i in inArray[:-1]]):
            raise ValueError("invalid input array")
        elif not all([len(i) == len(inArray) for i in inArray[:-1]]):
            raise ValueError("not square")

        aggCol  = DataFrame(inArray[-1]).T
        aggRow  = DataFrame([i[-1] for i in inArray[:-1]])
        if not aggCol.applymap(lambda x: x in (0,1)).all().all():
            raise ValueError("invalid data in column subtotal, must be 1 or 0")
        elif not aggRow.applymap(lambda x: x in (0,1)).all().all():
            raise ValueError("invalid data in row subtotals, must be 1 or 0")
        
        costDF  = DataFrame([i[:-1] for i in inArray[:-1]])

        for i in (costDF,aggCol,aggRow):
            i.columns = [j+1 for j in i.columns]
            i.index = [j+1 for j in i.index]

        return cls(costDF,aggCol,aggRow,dict())

    def __init__(self,costDF: DataFrame,aggCol: DataFrame,aggRow: DataFrame,BFS: Dict):
        self.costDF = costDF.copy()
        self.aggCol = aggCol.copy() # aggregate of Columns
        self.aggRow = aggRow.copy() # aggregate of Rows
        self.BFS    = BFS.copy()    #dict key is coord tuple,   value is assigned val

    def subtractMinimums(self,echo: bool=False) -> Hungarian:
        '''
        for each row, subtract minimum
        for each col, subtract minimum
        '''
        nCostDF = (self.costDF.copy()
                .apply(lambda x:x-min(x),axis=1)
                .apply(lambda x:x-min(x),axis=0))
        return Hungarian(nCostDF,self.aggCol,self.aggRow,self.BFS)

    def isOptimal(self) -> bool:
        '''
        if there is a feasible assignment with 0 entries of the zero entries,
        this assignment is optimal
        '''
        if self.BFS == dict():
            return False
        else:
            return True

    def coverLines(self, echo:str=False) -> Tuple[List[int],List[int]]:
        '''
        return a Tuple of (list of crossed rows, list of crossed cols)
        '''
        workingDF = self.costDF.copy()
        max0 = 1
        while max0 >0:
            rows = (workingDF == 0).sum(axis=1)
            cols = (workingDF == 0).sum(axis=0)
            max0 = max(rows.append(cols))

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
        # if echo:{display2(workingDF.applymap(lambda x:"-" if np.isposinf(x) else x))}
        rows = (workingDF == np.inf).all(axis=1)
        rows = rows[rows==True].index.tolist()
        cols = (workingDF == np.inf).all(axis=0)
        cols = cols[cols==True].index.tolist()
        return (rows,cols)

    def pivot(self, crossedLines:Tuple[List[int],List[int]],echo: bool=False):
        if self.isOptimal():
            # return copy if already optimal
            return self.copy()
        
        rows,cols = crossedLines
        maskedDF = self.costDF.copy()
        for row in rows:
            maskedDF.loc[row,:] = np.inf
        for col in cols:
            maskedDF.loc[:,col] = np.inf
        minVal = maskedDF.min().min()
        if echo:{display2(maskedDF.applymap(lambda x: "-" if x==np.inf else x))}
        workingDF = self.costDF.copy()
        for colIndex in workingDF.columns:
            for rowIndex in workingDF.index:
                if colIndex in cols and rowIndex in rows:
                    workingDF.loc[rowIndex,colIndex] += minVal
                elif colIndex not in cols and rowIndex not in rows:
                    workingDF.loc[rowIndex,colIndex] -= minVal
        if echo:{display2(workingDF)}

        base = Hungarian(workingDF,self.aggCol,self.aggRow,self.BFS)
        zeros = base.costDF[base.costDF == base.costDF.min().min()].stack().index.tolist()
        for zero in combinations(zeros,len(self.costDF)):
            x = set(i[0] for i in zero)
            y = set(i[1] for i in zero)
            if (len(x) == len(self.costDF) 
            and len(y) == len(self.costDF)):
                base.BFS = {(row,col):1 for row,col in zero}
        return base

    def solve(self,echo:bool=False):
        if self.isOptimal():
            return self
        else:
            return (self.subtractMinimums(echo=echo)
                .pivot(self.coverLines(echo=echo),echo=echo)
                .solve(echo=echo))

    # getters
    def to_LP(self,label:str="X"):
        
        def varRepr(row,col,label):
            return f"{label}_{row},{col}"

        # objective Function
        allVariables = [varRepr(row,col,label) for row in self.costDF.index for col in self.costDF.columns]
        objFunc = ObjectiveFunction.new("min",
            *[self.costDF.loc[row,col] for row in self.costDF.index for col in self.costDF.columns],
            labels=allVariables
        )

        rowConstraints = [Constraint.new(
            *[self.costDF.loc[row,col] for row in self.costDF.index],
            "<",self.aggCol.loc[1,col],
            labels = [varRepr(row,col,label) for row in self.costDF.index]
        ) for col in self.costDF.columns]
        
        colConstraints = [Constraint.new(
            *[self.costDF.loc[row,col] for col in self.costDF.columns],
            "<",self.aggRow.loc[row,1],
            labels = [varRepr(row,col,label) for col in self.costDF.columns]
        ) for row in self.costDF.index]

        nonNegConstraints = NonNeg.fromArray(*([">"]*len(allVariables)),labels=allVariables)

        return LP(
            objFunc,
            *rowConstraints,
            *colConstraints,
            *nonNegConstraints
            )

    def objectiveValue(self):
        '''
        return calculated objective value of this transport tableau using current assignments
        '''
        return sum([value * self.costDF.loc[row,col] for (row,col),value in self.BFS.items()])

    def bfsDF(self) -> DataFrame:
        baseDF = self.costDF.copy().applymap(lambda x:"")
        for (row,col),value in self.BFS.items():
            baseDF.loc[row,col] = value
        return baseDF

    def copy(self):
        return Hungarian(self.costDF,self.aggCol,self.aggRow,self.BFS)

    def __repr__(self):
        baseDF = self.costDF.join(Series(self.aggRow.loc[:,1],name="row")).append(Series(self.aggCol.loc[1,:],name="col")).fillna("")
        b = [f"X_{row},{col} = {c},  " for (row,col),c in self.BFS.items()]
        b = "".join(b)[:-1]
        return f"{baseDF}\nBFS :\n{self.bfsDF()}"
