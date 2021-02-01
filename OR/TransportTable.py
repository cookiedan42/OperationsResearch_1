import numpy as np
from pandas import DataFrame,Series
import pandas as pd

from .LP import LP
from .ObjectiveFunction import ObjectiveFunction
from .Constraint import Constraint
from .NonNeg import NonNeg
from .Format import Optimal, OptimalException

from typing import List,Tuple,Dict,TypeVar
TransportTable = TypeVar("TransportTable")

class TransportTable():

    @classmethod
    def new(cls,inArray:List[List[float]]) -> TransportTable:
        '''
        convert list of lists to DataFrame for TransportTable
        '''
        #data validity
        if not all([len(i)-1 == len(inArray[-1]) for i in inArray[:-1]]):
            raise ValueError("invalid input array")

        aggCol  = DataFrame(inArray[-1]).T
        aggRow  = DataFrame([i[-1] for i in inArray[:-1]])
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

    # LP stuff
    def to_LP(self,label: str = "X") -> LP:
        '''
        return the LP representing this tableau
        with factors as X_row,col
        '''
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
    
    #BFS stuff
    def northWest(self,echo: bool=False) -> TransportTable:
        '''
        assign a BFS using northWest Method
        '''
        def NW(row:int,col:int)-> Dict:
            '''
            private recursive function to do allocations
            '''
            if echo:
                df = (assignDF.join(Series(rowAgg, name="remain"))
                      .append(Series(colAgg, name="remain")).fillna(""))
                try:
                    display(df)
                except:
                    print(df)

            # BFS complete
            if len(BFS) == len(colAgg) + len(rowAgg) - 1:
                return None
            
            # allocate to current checking
            minVal = min(colAgg.loc[col],rowAgg.loc[row])
            assignDF.loc[row,col] = minVal
            BFS[(row,col)] = minVal
            colAgg.loc[col] -= minVal
            rowAgg.loc[row] -= minVal

            # go to next
            if colAgg.loc[col] == 0 and col < len(assignDF.columns):
                colAgg.loc[col] = "X"
                return NW(row,col+1 )
            else: # rowAgg.loc[row] == 0 and row < len(assignDF.index)
                rowAgg.loc[row] = "X"
                return NW(row+1,col)

        assignDF = self.costDF.copy().applymap(lambda x:"")
        colAgg = self.aggCol.copy().loc[1,:]
        rowAgg = self.aggRow.copy().loc[:,1]
        BFS = dict()
        NW(1,1)
        
        return TransportTable(self.costDF,self.aggCol,self.aggRow,BFS)

    def minimumCost(self,echo: bool=False) -> TransportTable:
        '''
        assign a BFS using minimum cost Method
        '''
        def MC():
            '''
            private recursive function to do allocations
            '''
            if echo:
                df = (assignDF
                    .join(Series(rowAgg,name="remain"))
                    .append(Series(colAgg,name="remain"))
                    .fillna("")
                )
                try:
                    display(df)
                except:
                    print(df)

            if len(BFS) == len(colAgg) + len(rowAgg) - 1:
                #BFS complete
                return None

            # find smallest cost
            row,col = tCostDF[tCostDF == tCostDF.min().min()].stack().index.tolist()[0]
            
            
            # allocate 
            minVal = min(rowAgg.loc[row],colAgg.loc[col])
            assignDF.loc[row,col] = minVal
            BFS[(row,col)] = minVal
            colAgg.loc[col] -= minVal
            rowAgg.loc[row] -= minVal
          
            # cross out row/col by changing it to np.inf
            if colAgg.loc[col] == 0:
                tCostDF.loc[:,col] = np.inf
                colAgg.loc[col] = "X"
                return MC()
            elif rowAgg.loc[row] == 0:
                tCostDF.loc[row,:] = np.inf
                rowAgg.loc[row] = "X"
                return MC()

        tCostDF = self.costDF.copy()
        assignDF = self.costDF.copy().applymap(lambda x:"")
        colAgg = self.aggCol.copy().loc[1,:]
        rowAgg = self.aggRow.copy().loc[:,1]
        BFS = dict()
        MC()
        return TransportTable(self.costDF,self.aggCol,self.aggRow,BFS)
    
    def vogel(self,echo: bool=False) -> TransportTable:
        '''
        assign a BFS using vogel method
        '''
        def oppCost(series:Series) -> float:
            s = sorted(series)
            return s[1] - s[0]
        def VG():
            '''
            private recursive function to do allocations
            '''
            if echo:
                df = (assignDF
                    .join(Series(rowAgg,name="remain"))
                    .append(Series(colAgg,name="remain"))
                    .fillna("")
                )
                try:
                    display(df)
                except:
                    print(df)
            
            if len(BFS) == len(colAgg) + len(rowAgg) - 1:
                #BFS complete
                return None

            # select grid to maximise
            rowOpp = [oppCost(tCostDF.loc[i,:]) if not rowAgg[i]=="X" else 0 for i in rowAgg.index]
            colOpp = [oppCost(tCostDF.loc[:,i]) if not colAgg[i]=="X" else 0 for i in colAgg.index]
            maxOpp = max(rowOpp + colOpp)
            if maxOpp in colOpp:
                col = colOpp.index(maxOpp) + 1
                row = tCostDF.loc[:,col].idxmin()
            else: # maxOpp in rowOpp
                row = rowOpp.index(maxOpp) + 1
                col = tCostDF.loc[row,:].idxmin()

            # allocate
            minVal = min(rowAgg.loc[row],colAgg.loc[col])
            assignDF.loc[row,col] = minVal
            BFS[(row,col)] = minVal
            colAgg.loc[col] -= minVal
            rowAgg.loc[row] -= minVal
            

            # cross out the row/col
            if colAgg.loc[col] == 0:
                tCostDF.loc[:,col] = np.inf
                colAgg.loc[col] = "X"
                return VG()
            elif rowAgg.loc[row] == 0:
                tCostDF.loc[row,:] = np.inf
                rowAgg.loc[row] = "X"
                return VG()


        tCostDF = self.costDF.copy()
        assignDF = self.costDF.copy().applymap(lambda x:"")
        rowAgg = self.aggRow.copy().loc[:,1]
        colAgg = self.aggCol.copy().loc[1,:]
        BFS = dict()
        VG()
        return TransportTable(self.costDF,self.aggCol,self.aggRow,BFS)

    # solving
    def getEnteringVar(self,echo=False) -> Tuple[int,int]:
        '''
        find the entering var
        '''
        variables = [f"U{i}" for i in self.costDF.index]+[f"V{i}" for i in self.costDF.columns]
        LHS = DataFrame(columns=variables)
        body = [DataFrame([[1]],columns=[LHS.columns[0]])]
        body += [DataFrame([[1,1]],columns=[f"U{row}",f"V{col}"]) for (row,col),c in self.BFS.items()]
        
        LHS = LHS.append(pd.concat(body,sort=False),sort=False).fillna(0)
        LHS.index = variables
        RHS = DataFrame([0]+[self.costDF.loc[row,col] for (row,col),v in self.BFS.items()],columns=["RHS"])
        RHS.index = variables
        soln = np.linalg.solve(LHS,RHS)
        soln = Series([i[0] for i in soln],index=variables)
        
        if echo:
            print("From complementary slackness property:")
            for i in LHS.index:
                row = LHS.loc[i]
                out = "".join([f" + {i}" for i in variables if row[i] == 1]) + f" = {RHS.loc[i,'RHS']}"
                print(out[3:])
            print("\nA solution is:")
            for i in soln.index:
                print(f"{i} = {soln[i]}")
            print("\nComputing Ui + Vj – Cij for the nonbasic variables:")

        enteringVar = (None,0)
        for row in self.costDF.index:
            for col in self.costDF.columns:
                if (row,col) in self.BFS.keys():
                    continue
                calc = soln[f"U{row}"] + soln[f"V{col}"] - self.costDF.loc[row,col]
                if echo:{print(f"U{row} + V{col} - C{row}{col} : X{row}{col} = {calc}")}
                if calc >= enteringVar[1]:
                    enteringVar = ((row,col),calc)

        if enteringVar[0] == None:
            # optimal solution, no complementary slackness
            raise OptimalException("Optimal Solution, no Entering Variable")
        if echo:{print(f"\nEntering Variable is X_{row},{col}")}
        return enteringVar[0]

    def findLoop(self,start: Tuple[int,int],echo=False) -> List[Tuple[int,int]]:
        '''
        recursively find _a_ valid closed loop
        initial path is [(startRow,startCol)]
        '''
        def FL(path: Tuple[Tuple[int,int]],axis="col") -> Tuple[Tuple[int,int]]:
            if ( len(path) > 3 and path[-1] == path[0]):
                if echo:{print(f"Pivot on loop of {path[:-1]}")}
                return path[:-1] # drop the repeated start

            prevRow,prevCol = path[-1]
            if axis == "col":
                nextSteps = tuple((i,prevCol) for i in self.costDF.index)
                nextSteps = tuple(i for i in nextSteps if i not in path[1:])
                nextSteps = tuple(i for i in nextSteps if i in list(self.BFS.keys()) or i == path[0])
                out =  tuple(FL(path+(i,),axis="row") for i in nextSteps)
            else: #axis = "row"
                nextSteps = tuple((prevRow,i) for i in self.costDF.columns)
                nextSteps = tuple(i for i in nextSteps if i not in path[1:])
                nextSteps = tuple(i for i in nextSteps if i in list(self.BFS.keys()) or i == path[0])
                out = tuple(FL(path+(i,),axis="col") for i in nextSteps)
            out = tuple(i for i in out if i!=None)
            if len(out) == 1:
                return out[0]
            elif len(out) > 1:
                return out
        
        return FL((start,))

    def loopPivot(self,loopSeq,echo=False):
        '''
        do the Pivot using loopSeq
        '''
        evenCells = ()  # contain tuples of indexes of even cells
        oddCells = ()   # contain tuples of indexes of odd cells
        i = 0
        while i < len(loopSeq):
            evenCells += (loopSeq[i],)
            oddCells  += (loopSeq[i+1],)
            i+=2
        newBFS = self.BFS.copy()
        newBFS[loopSeq[0]] = 0
        oddVals = [newBFS[i] for i in oddCells]
        theta = min(oddVals)
        leavingVariable = oddCells[oddVals.index(theta)]       
        #TODO potential degeneracy when mutiple leaving variables can exist?
        
        if echo:{print(f"Leaving Variable is: X_{leavingVariable[0]},{leavingVariable[1]}")}

        for i in oddCells:
            newBFS[i] -= theta
        for i in evenCells:
            newBFS[i] += theta
        del newBFS[leavingVariable]
        return TransportTable(self.costDF,self.aggCol,self.aggRow,newBFS)

    def solve(self,echo: bool=False):
        '''
        solve the transport tableau
        '''
        newTab = self
        while True:
            try:
                enteringVar = newTab.getEnteringVar(echo=echo)
            except OptimalException:
                return newTab
            loop = newTab.findLoop(enteringVar,echo=echo)
            newTab = newTab.loopPivot(loop,echo=echo)
            if echo:{print("-"*20)}
        #TODO check for degenerate looping

    def isOptimal(self) -> bool:
        '''
        check if this is optimal Tableau
        '''
        try:
            self.getEnteringVar()
        except OptimalException:
            return True
        return False

    # getters
    def objectiveValue(self):
        '''
        return calculated objective value of this transport tableau using current assignments
        '''
        return sum([value * self.costDF.loc[row,col] for (row,col),value in self.BFS.items()])
            
    def bfsDF(self) -> DataFrame:
        baseDF = self.costDF.copy().applymap(lambda x:"")
        for (row,col),value in self.BFS:
            baseDF.loc[row,col] = value
        return baseDF

    def __repr__(self):
        baseDF = self.costDF.join(Series(self.aggRow.loc[:,1],name="row")).append(Series(self.aggCol.loc[1,:],name="col")).fillna("")
        b = [f"X_{row},{col} = {c},  " for (row,col),c in self.BFS.items()]
        b = "".join(b)[:-1]
        return f"{baseDF}\nBFS : {b}"
