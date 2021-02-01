from pandas import DataFrame
import pandas as pd
import numpy as np

from .Format import OptimalException, display2
from .LP import LP
from .NonNeg import NonNeg

from typing import List,Tuple,Dict,TypeVar
TableauBase = TypeVar("TableauBase")

class TableauBase():
    @classmethod
    def fromLP(cls,inLP:LP) -> TableauBase:
        inLP = inLP.standardForm()
        constraints = [(i+1,c.add_S(i+1)) for i,c in enumerate(inLP.constraints) if not isinstance(c,NonNeg)]
        target =  inLP.objFunc.toDF().columns[0]
        label = inLP.objFunc.label

        uniqueVars = set()
        uniqueVars.update(inLP.objFunc.get_variables())
        for i,c in constraints:
            uniqueVars.update(c.get_variables())

        baseDF = pd.concat( [inLP.objFunc.toDF()] + [c.toDF(i) for i,c in constraints] ,sort=False)
        baseDF = baseDF.fillna(0).drop(columns="sign")
        #add Basic var Column

        basicVar = [""]
        for i in range(len(constraints)):
            if f"A{i+1}" in uniqueVars:
                basicVar += [f"A{i+1}"]
            elif f"S{i+1}" in uniqueVars:
                basicVar += [f"S{i+1}"]
            else:
                basicVar += [f"{label}{i+1}"]
        basicVar = pd.DataFrame( basicVar,columns=["Basic"])

        cols = ([target]+# if inLP.objFunc.sign == "max" else f"-{target}"]+
            sorted([i for i in uniqueVars if label in i])+
            sorted([i for i in uniqueVars if "S" in i])+
            sorted([i for i in uniqueVars if "A" in i])+
            ["RHS"])

        return TableauBase(baseDF[cols],basicVar)

    def __init__(self,dataFrame: DataFrame,basicVar: DataFrame):
        self.dataFrame = dataFrame
        self.basicVar = basicVar

    def canonical(self,echo:bool = False) -> TableauBase:
        return self.setBFS(list(self.basicVar.loc[:,"Basic"])[1:],echo=echo)

    def asDF(self) -> DataFrame:
        return self.dataFrame.join(self.basicVar)

    def __repr__(self) -> str:
        return str(self.asDF())

    def pivot(self, factor: str, row: int,echo: bool=False) -> TableauBase:
        '''
        core pivot action using matrix math
        '''
        row = int(row)
        pivot_num = self.dataFrame[factor][row]
        pivot_col = self.dataFrame[factor]
        pivot_row = self.dataFrame.iloc[row]

        enter_row = pivot_row / pivot_num
        result    = self.dataFrame - np.dot(DataFrame(pivot_col),DataFrame(enter_row).T)
        result.iloc[row] = enter_row

        newVar = self.basicVar.copy()
        newVar.loc[row] = factor

        out = TableauBase(result,newVar)
        if echo:{display2(out)}
        return out

    def getBFS(self) -> Dict[str,float]:
        return {k:v for k,v in zip(self.basicVar.loc[1:,"Basic"],self.dataFrame.loc[1:,"RHS"])}

    def getObjectiveVal(self) -> Dict[str,float]:
        if self.dataFrame.columns[0][0] == "-":
            return {self.dataFrame.columns[0][1:] : -1* self.dataFrame.loc[0,"RHS"]}
        else:
            return {self.dataFrame.columns[0] : self.dataFrame.loc[0,"RHS"]}

    def setBFS(self,bfs:List[str],echo=False) -> TableauBase:
        '''
        pivot tableau to use new BFS
        '''
        k = self
        if echo:{display2(k)}
        for i,variable in enumerate(bfs):
            k = k.pivot(variable,i+1,echo=echo)
        return k

    def isOptimal(self) -> bool:
        if all(self.dataFrame.loc[0].drop("RHS") >= 0):
            return True
        else:
            return False

    def pickPivotCol(self,echo=False) -> str:
        '''
        get the most negative column
        '''
        minVal = self.dataFrame.loc[0].iloc[1:-1].idxmin()
        if echo:{display2(DataFrame(self.dataFrame.loc[0]).T)}
        if self.dataFrame.loc[0,minVal] < 0 :
            return minVal
        else:
            raise OptimalException("row 0 all positive, Tableau is optimal")
            # return None

    def pickPivotRow(self,factor: str,echo=False) -> int:
        '''
        ratio test
        1) solution divide entering col must be positive
        2) eliminate all negatives and infs
        3) take smallest of all remaining positive non inf
        4) --> such that pivoting on this will cause next solution set to remain positive no.
        '''
        col = self.dataFrame.loc[:,factor]
        RHS = self.dataFrame.loc[:,"RHS"]
        
        l = RHS/col
        l = l[1:][l > 0][l != np.nan]
        if echo:{display2(self.asDF().join(DataFrame(l,columns=["Ratio Test"])))}
        if len(l) >= 1:
            return l.idxmin()
        else:
            raise OptimalException("no valid pivot row found, Tableau is optimal")

    def pickPivot(self,echo=False) -> Tuple[str,str]:
        '''
        return (row,column)
        combining pickRow and pickCol
        '''
        try:
            col = self.pickPivotCol(echo=echo)
            row = self.pickPivotRow(col,echo=echo)
            return col,row
        except Exception as e:
            raise e

    def autoPivot(self,echo=False) -> TableauBase:
        '''
        stack a search and a pivot together, single cycle
        '''
        return self.pivot(*self.pickPivot(echo=echo),echo=echo)

    def solve(self,echo=False,past=[]) -> TableauBase:
        '''
        loop autosolve until result produced
        or error encountered
        '''
        k = self
        try:
            while True:
                k = k.autoPivot(echo=echo)
                if echo:{display2(k.asDF())}
        except Exception as e:
            if echo:{print(e)}
        finally:
            return k
