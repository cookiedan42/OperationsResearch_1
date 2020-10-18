from numpy.matrixlib.defmatrix import matrix
from .Format import *
from .LP import LP
import numpy as np
from pandas import DataFrame,Series
import pandas as pd


class TransportTable():
    @classmethod
    def new(cls,costs,RHS,BHS,RHSName = "Remain",BHSName = "Remain"):
        costDF = DataFrame(costs).copy()
        assignDF = DataFrame(np.zeros_like(costDF)).copy()
        for df in [costDF,assignDF]:
            df.columns = [i+1 for i in range(len(df.columns))]
            df.index = [i+1 for i in range(len(df.index))]
        
        RHS = Series(RHS,name=RHSName).copy()
        RHS.index = RHS.index = [i+1 for i in range(len(RHS.index))]
        BHS = Series(BHS,name=BHSName).copy()
        BHS.index = BHS.index = [i+1 for i in range(len(BHS.index))]
        return TransportTable(costDF,assignDF,RHS,BHS)

    def __init__(self,costDF,assignDF,RHS,BHS):
        self.costDF = costDF.copy()
        self.assignDF = assignDF.copy()
        self.BHS = BHS.copy()
        self.RHS = RHS.copy()

    # table stuff
    def assign(self,rowIndex,colIndex,amount):
        if (amount<= self.BHS.loc[colIndex]
        and amount<= self.RHS.loc[rowIndex]):

            newRHS = self.RHS.copy()
            newRHS.loc[rowIndex] -= amount

            newBHS = self.BHS.copy()
            newBHS.loc[colIndex] -= amount

            newAssignDF = self.assignDF.copy()
            newAssignDF.loc[rowIndex,colIndex] += amount

            return TransportTable(self.costDF.copy(),newAssignDF, newRHS,newBHS)
        else:
            raise ValueError("too large of an assignment") 
    def assignMax(self,rowIndex,colIndex):
        amount = min(self.RHS.loc[rowIndex],self.BHS.loc[colIndex])
        return self.assign(rowIndex,colIndex,amount)

    # LP stuff 
    def to_LP(self,):
        variablesNo = self.costDF.shape[0]*self.costDF.shape[1]
        #objectiveFn
        objFn = ["min"] + list(self.costDF.to_numpy().flatten())
        signs = [">"] * variablesNo
        factorNames = []
        constraints = []

        #row constraints
        for rowIndex in self.costDF.index:
            template = [0]*variablesNo + ["=",self.RHS.loc[rowIndex]]
            for colIndex in self.assignDF.columns:
                factorNames +=[f"X{subscript(rowIndex)},{subscript(colIndex)}"]
                template[(rowIndex-1)*self.assignDF.shape[1] + colIndex-1] = 1
            constraints.append(template)
        
        #col constraints
        for colIndex in self.assignDF.columns:
            template = [0]*variablesNo + ["=",self.BHS.loc[colIndex]]
            for rowIndex in self.assignDF.index:       
                template[(rowIndex-1)*self.assignDF.shape[1] + colIndex-1] = 1
            constraints.append(template)
        return LP.new(objFn,constraints,signs,factorNames=factorNames)
    def to_DualLP(self):
        baseLP = self.to_LP()
        factorNames = [f"u{i+1}" for i in range(self.assignDF.shape[0])] + [f"v{i+1}" for i in range(self.assignDF.shape[1])]
        return baseLP.getDual(factorNames=factorNames)

    #BFS
    def northWest(self,echo=False,origBHS=False,origRHS=False):
        if echo == True:
            self.display()
        if self.RHS.sum() + self.BHS.sum() ==0:
            #all assignments done
            return TransportTable(self.costDF.copy(),self.assignDF.copy(),origRHS,origBHS)
        if type(origBHS) == bool:
            # only copy on first loop
            # otherwise pass through orig RHS and BHS
            origBHS = self.BHS.copy()
            origRHS = self.RHS.copy()


        rowIndex = -1
        colIndex = -1
        for RHSVal in self.RHS:
            if RHSVal!=0:
                rowIndex = 1+ self.RHS.to_list().index(RHSVal)
                break
        for BHSVal in self.BHS:
            if BHSVal!=0:
                colIndex = 1+self.BHS.to_list().index(BHSVal)
                break
        return (self.assignMax(rowIndex,colIndex)
                    .northWest(echo=echo,origBHS=origBHS,origRHS=origRHS))
    
    def minimumCost(self,echo=False,prevCost=0,origBHS=False,origRHS=False):
        if echo == True:
            self.display()
        if self.RHS.sum() + self.BHS.sum() ==0:
            #all assignment done
            return TransportTable(self.costDF.copy(),self.assignDF.copy(),origRHS,origBHS)
        if type(origBHS) == bool:
            # only copy on first loop
            # otherwise pass through orig RHS and BHS
            origBHS = self.BHS.copy()
            origRHS = self.RHS.copy()

        costs =  np.sort(np.unique(self.costDF.to_numpy().flatten()))
        filter_arr = costs > prevCost-1
        costs = costs[filter_arr]
        # start searching from equal to previous cost
        for cost in costs:
            for rowIndex in self.costDF.index:
                for colIndex in self.costDF.columns:
                    if( self.costDF.loc[rowIndex,colIndex] == cost
                    and self.RHS.loc[rowIndex] != 0
                    and self.BHS.loc[colIndex] != 0):
                        return (self.assignMax(rowIndex,colIndex)
                                    .minimumCost(echo=echo,prevCost=cost,origBHS=origBHS,origRHS=origRHS))
        
        raise RuntimeWarning("minimum Cost fell through, likely casued by an inbalanced Transport Table")
    
    def vogel(self,echo=False,origBHS=False,origRHS=False):
        if echo:
            self.display()
        if self.RHS.sum() + self.BHS.sum() ==0:
            #all assignment done
            return TransportTable(self.costDF.copy(),self.assignDF.copy(),origRHS,origBHS)
        if type(origBHS) == bool:
            # only copy on first loop
            # otherwise pass through orig RHS and BHS
            origBHS = self.BHS.copy()
            origRHS = self.RHS.copy()
        
        RHSOppCost = []
        for i in range(len(self.RHS)):
            rowArr = np.sort(np.unique(self.costDF.iloc[i,:]))
            if self.RHS.iloc[i]==0 or len(rowArr)==1:
                RHSOppCost += [0]
            else:
                RHSOppCost +=[rowArr[1] - rowArr[0]]

        BHSOppCost = []
        for i in range(len(self.BHS)):
            colArr = np.sort(np.unique(self.costDF.iloc[:,i]))
            if self.BHS.iloc[i]==0 or len(colArr)==1:
                BHSOppCost += [0]
            else:
                BHSOppCost +=[colArr[1] - colArr[0]]

        maxCost = max(RHSOppCost + BHSOppCost)
        if maxCost in RHSOppCost:
            rowIndex = RHSOppCost.index(maxCost)+1
            toSearch = self.costDF.loc[rowIndex].copy()
            for i in self.BHS.index:
                if self.BHS.loc[i] == 0:
                    toSearch.loc[i] = np.inf
            for i in self.BHS.index:
                if toSearch.loc[i] == min(toSearch):
                    return (self.assignMax(rowIndex,i)
                                .vogel(echo=echo,origRHS=origRHS,origBHS=origBHS))
        else: #maxCost in BHSOppCost
            colIndex = BHSOppCost.index(maxCost)+1
            toSearch = self.costDF.loc[:,colIndex].copy()
            for i in self.RHS.index:
                if self.RHS.loc[i] == 0:
                    toSearch.loc[i] = np.inf
            for i in self.RHS.index:
                if toSearch.loc[i] == min(toSearch):
                    return (self.assignMax(i,colIndex)
                                .vogel(echo=echo,origRHS=origRHS,origBHS=origBHS))

        raise RuntimeWarning("Vogel fell through, likely casued by an inbalanced Transport Table")

    # solving
    def solve(self,echo=False):
        if echo:
            self.display()
        eV = self.getEnteringVar(echo=echo)
        if eV == False: # is optimal to not run the calculation twice
            return self
        loop = self.findLoop((eV,),echo=echo)

        #shit breaks down here



        return self.loopPivot(loop,echo=echo).solve(echo=echo)
    
    def isOptimal(self):
        # reuse duality code here
        return bool(self.getEnteringVar)


    def getEnteringVar(self,echo=False):
        workingAssign = DataFrame(np.zeros_like(self.assignDF))
        workingAssign.columns = self.assignDF.columns
        workingAssign.index = self.assignDF.index



        # SOLVE FOR U and V
        
        matrixRHS = []
        LHSIndex = Series(self.RHS.index).copy().apply(lambda x:"U" + str(x))
        LHSIndex = LHSIndex.append(Series(self.BHS.index).copy().apply(lambda x:"V" + str(x)))
        LHS_Template = Series([0]*len(LHSIndex),index=LHSIndex)
        matrixLHS = DataFrame(columns=LHSIndex)

        for colIndex in self.BHS.index:
            for rowIndex in self.RHS.index:
                if self.assignDF.loc[rowIndex,colIndex] != 0:
                    LHSrow = LHS_Template.copy()
                    LHSrow.loc["U" + str(rowIndex)] = 1
                    LHSrow.loc["V" + str(colIndex)] = 1
                    matrixLHS = matrixLHS.append(LHSrow,ignore_index=True)
                    matrixRHS += [self.costDF.loc[rowIndex,colIndex]]
        matrixRHS = Series(matrixRHS)

        toDrop = ()
        # drop 0 costs
        rowCost = (self.costDF != 0).astype(int).sum(axis=1)
        colCost = (self.costDF != 0).astype(int).sum(axis=0)
        for i in rowCost.index:
            if rowCost.loc[i] == 0:
                toDrop+=(f"U{i}",)

        for i in colCost.index:
            if colCost.loc[i] == 0:
                toDrop+=(f"V{i}",)

        for i in toDrop:
            matrixLHS = matrixLHS.drop(columns=i)

        while matrixLHS.shape[0] < matrixLHS.shape[1]:
            toDrop += (f"{matrixLHS.columns[0]}",)
            matrixLHS = matrixLHS.drop(columns=toDrop[-1])

        matrixLHS = matrixLHS.applymap(lambda x:float(x))
        soln = np.linalg.solve(matrixLHS,matrixRHS)
        soln = Series(soln,index=matrixLHS.columns)
        soln = soln.append(Series([0]*len(toDrop),index=toDrop))
        soln = soln[LHSIndex]

        for i in rowCost.index:
            if rowCost.loc[i] == 0:
                workingAssign = workingAssign.drop(index=i)

        for i in colCost.index:
            if colCost.loc[i] == 0:
                workingAssign = workingAssign.drop(columns=i)


        basicVarVals = dict()
        for colIndex in workingAssign.columns:
            for rowIndex in workingAssign.index:
                uVal = soln.loc[f'U{rowIndex}']
                vVal = soln.loc[f'V{colIndex}']
                basicVarVals[f"U{subscript(rowIndex)}"] = uVal
                basicVarVals[f"V{subscript(colIndex)}"] = vVal

                if self.assignDF.loc[rowIndex,colIndex] == 0:
                    if echo:{print(f"U{subscript(rowIndex)} + V{subscript(colIndex)} - C{subscript(rowIndex)},{subscript(colIndex)} = {uVal + vVal - self.costDF.loc[rowIndex,colIndex]}")}
                    workingAssign.loc[rowIndex,colIndex] = soln.loc[f"U{rowIndex}"] + soln.loc[f"V{colIndex}"] - self.costDF.loc[rowIndex,colIndex]
        if echo:{print(basicVarVals)}


        
        maxW = workingAssign.max().max()
        
        if maxW == 0:
            if echo:{print(f"all W values are <= 0, optimal solution found\nobjective value is {self.objectiveValue()}\n")}
            return False
        
        for colIndex in workingAssign.columns:
            for rowIndex in workingAssign.index:
                if workingAssign.loc[rowIndex,colIndex] == maxW:
                    if echo:{print(f"Entering Variable is {(rowIndex,colIndex)}\n")}
                    return (rowIndex,colIndex)

    def findLoop(self,path,axis="col",echo=False):
        # path is a tuple of 
            # tuples of (row,col)
        
        #base case when loop closes
        if ( len(path) > 3
        and path[-1] == path[0]):
            if echo:{print(f"Pivot on loop of {path[:-1]}")}
            return path[:-1] # drop the repeated start
        prevRow,prevCol = path[-1]

        if axis == "col":
            nextSteps = tuple((i,prevCol) for i in self.RHS.index)
            nextSteps = tuple(i for i in nextSteps if i not in path[1:])
            nextSteps = tuple(i for i in nextSteps if self.assignDF.loc[i[0],i[1]] != 0 or i == path[0])
            out =  tuple(self.findLoop(path+(i,),axis="row",echo=echo) for i in nextSteps)
        else: #axis = "row"
            nextSteps = tuple((prevRow,i) for i in self.BHS.index)
            nextSteps = tuple(i for i in nextSteps if i not in path[1:])
            nextSteps = tuple(i for i in nextSteps if self.assignDF.loc[i[0],i[1]] != 0 or i == path[0])
            out = tuple(self.findLoop(path+(i,),axis="col",echo=echo) for i in nextSteps)

        out = tuple(i for i in out if i!=None)

        if len(out) == 1:
            return out[0]
        elif len(out) > 1:
            return out
    
    def loopPivot(self,loopSeq,echo=False):
        '''
        given that we have somehow found the even and odd squares,
        now we run the loopPivot on the Transport Table
        '''
        if echo:{self.display()}
        evenIndexTuple = ()
        oddIndexTuple = ()
        i = 0
        while i < len(loopSeq):
            evenIndexTuple += (loopSeq[i],)
            oddIndexTuple  += (loopSeq[i+1],)
            i+=2
        newAssignDF = self.assignDF.copy()
        # evenIndexTuple is an array of even squares
        # oddIndexTuple is an array of odd squares
        # members are tuples of (row,col) similar to x(row,col) notation
        evenVals = [newAssignDF.loc[i[0],i[1]] for i in evenIndexTuple]
        oddVals = [newAssignDF.loc[i[0],i[1]] for i in oddIndexTuple]
        theta = min(oddVals)
        leavingVariable = oddIndexTuple[oddVals.index(theta)]

        if echo:{print(f"Leaving Variable is: X{leavingVariable}")}

        for i in oddIndexTuple:
            newAssignDF.loc[i[0],i[1]] -= theta
        for i in evenIndexTuple:
            newAssignDF.loc[i[0],i[1]] += theta
        return TransportTable(self.costDF.copy(),newAssignDF,self.RHS.copy(),self.BHS.copy())

    # getters
    def objectiveValue(self):
        '''
        return calculated objective value of this transport tableau using current assignments
        '''
        return self.costDF.mul(self.assignDF).sum().sum()

    #display section
    def displayFormat(self):
        dispDF =self.assignDF.copy()
        dispDF.insert(self.assignDF.shape[1],self.RHS.name,self.RHS)
        dispColVars = self.BHS.copy()
        dispColVars = dispColVars.append(Series([""],index=[self.RHS.name]))
        dispColVars.name = self.BHS.name
        dispDF = dispDF.append(dispColVars)
        return dispDF
    
    def display(self):
        displayHelper(self.displayFormat())
    
    def __repr__(self):
        return str(self.displayFormat())