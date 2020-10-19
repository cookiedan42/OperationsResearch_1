from .Format import *
from .LP import LP
import numpy as np
from pandas import DataFrame,Series

class TransportTable():
    @classmethod
    #factory to produce BFS-less tptTable
    def new(cls,costs,RHS,BHS,RHSName = "RHS",BHSName = "BHS"):
        costDF = DataFrame(costs).copy()
        costDF.columns = [i+1 for i in range(len(costDF.columns))]
        costDF.index = [i+1 for i in range(len(costDF.index))]
        
        RHS = Series(RHS,name=RHSName).copy()
        RHS.index = [i+1 for i in range(len(RHS.index))]
        BHS = Series(BHS,name=BHSName).copy()
        BHS.index = [i+1 for i in range(len(BHS.index))]
        BFS = dict()
        return TransportTable(costDF,BFS,RHS,BHS)

    def __init__(self,costDF,BFS,RHS,BHS):
        self.costDF = costDF.copy()
        self.BFS = BFS  #dict key is coord tuple,   value is assigned val
        self.BHS = BHS.copy()
        self.RHS = RHS.copy()

    # table stuff
    def assign(self,rowIndex,colIndex,amount):  # TODO
        if (amount <= self.BHS.loc[colIndex]     # amount + col sum
        and amount <= self.RHS.loc[rowIndex]):   # amount + row sum

            newRHS = self.RHS.copy()
            newRHS.loc[rowIndex] -= amount

            newBHS = self.BHS.copy()
            newBHS.loc[colIndex] -= amount

            BFS = self.BFS.copy()
            BFS[(rowIndex,colIndex)] = amount

            return TransportTable(self.costDF.copy(),BFS, newRHS,newBHS)
        else:
            raise ValueError("too large of an assignment") 
    def assignMax(self,rowIndex,colIndex):
        amount = min(self.RHS.loc[rowIndex],self.BHS.loc[colIndex])
        return self.assign(rowIndex,colIndex,amount)

    # LP stuff 
    def to_LP(self):
        variablesNo = self.costDF.shape[0]*self.costDF.shape[1]
        #objectiveFn
        objFn = ["min"] + list(self.costDF.to_numpy().flatten())
        signs = [">"] * variablesNo
        factorNames = []
        constraints = []

        #row constraints
        for rowIndex in self.costDF.index:
            template = [0]*variablesNo + ["=",self.RHS.loc[rowIndex]]
            for colIndex in self.costDF.columns:
                factorNames +=[f"X{subscript(rowIndex)},{subscript(colIndex)}"]
                template[(rowIndex-1)*self.costDF.shape[1] + colIndex-1] = 1
            constraints.append(template)
        
        #col constraints
        for colIndex in self.costDF.columns:
            template = [0]*variablesNo + ["=",self.BHS.loc[colIndex]]
            for rowIndex in self.costDF.index:       
                template[(rowIndex-1)*self.costDF.shape[1] + colIndex-1] = 1
            constraints.append(template)
        return LP.new(objFn,constraints,signs,factorNames=factorNames)

    def to_DualLP(self):
        baseLP = self.to_LP()
        factorNames = [f"u{i}" for i in self.costDF.index] + [f"v{i}" for i in self.costDF.columns]
        return baseLP.getDual(factorNames=factorNames)

    #BFS

    def northWest(self,echo=False,origBHS=False,origRHS=False,curDir="col",checking=(1,1)):
        #checking is tuple of coordinates
        if echo == True:
            self.display()
        if self.RHS.sum() + self.BHS.sum() == 0:
            #all assignments done
            return TransportTable(self.costDF.copy(),self.BFS.copy(),origRHS,origBHS)
        if type(origBHS) == bool:
            # only copy on first loop
            # otherwise pass through orig RHS and BHS
            origBHS = self.BHS.copy()
            origRHS = self.RHS.copy()
        
        if curDir =="col":
            nextState = self.assignMax(checking[0],checking[1])
            rowSum = origRHS.loc[checking[0]]
            for k,v in  nextState.BFS.items():
                if k[0] == checking[0]:
                    rowSum -= v
            if rowSum == 0:
                # row Cleared, assign next row
                return nextState.northWest(echo=echo,origBHS=origBHS,origRHS=origRHS,curDir="row",checking=(checking[0]+1,checking[1]))
            else:
                # col uncleared assign next col
                return nextState.northWest(echo=echo,origBHS=origBHS,origRHS=origRHS,curDir="col",checking=(checking[0],checking[1]+1))

        else: # curDir = "row"
            # check (checking[0]+1,checking[1])
            nextState = self.assignMax(checking[0],checking[1])
            colSum = origBHS.loc[checking[1]]
            for k,v in  nextState.BFS.items():
                if k[1] == checking[1]:
                    colSum -= v
            if colSum == 0:
                # col Cleared, assign next col
                return nextState.northWest(echo=echo,origBHS=origBHS,origRHS=origRHS,curDir="col",checking=(checking[0],checking[1]+1))
            else:
                # col uncleared assign next row
                return nextState.northWest(echo=echo,origBHS=origBHS,origRHS=origRHS,curDir="row",checking=(checking[0]+1,checking[1]))
    
    def minimumCost(self,echo=False,prevCost=0,origBHS=False,origRHS=False,crossedOut=[[],[]]):
        if echo == True:
            self.display()
        if self.RHS.sum() + self.BHS.sum() == 0:
            #all assignment done
            if len(self.BFS) + 1 == len(self.RHS) + len(self.BHS):
            # if enough BFS found
                return TransportTable(self.costDF.copy(),self.BFS.copy(),origRHS,origBHS)
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
                if rowIndex in crossedOut[0]:
                        continue
                for colIndex in self.costDF.columns:
                    if colIndex in crossedOut[1]:
                        continue
                    if(self.costDF.loc[rowIndex,colIndex] == cost):
                        nextState = self.assignMax(rowIndex,colIndex)
                        
                        rowSum = origRHS.loc[rowIndex]
                        colSum = origBHS.loc[colIndex]
                        for k,v in  nextState.BFS.items():
                            if k[0] == rowIndex:
                                rowSum -= v
                            if k[1] == colIndex:
                                colSum -= v
                        #only cross out one row/col
                        if (rowSum == 0 
                        and len(self.RHS) - len(crossedOut[0]) > 1):
                            crossedOut[0].append(rowIndex)
                        elif (colSum == 0                           # TODO testing if last case assigned is value 0 works
                        and len(self.BHS)-len(crossedOut[1]) > 1):  # len-len is supposed to not eliminate the last in each dir
                            crossedOut[1].append(colIndex)
                        return (nextState.minimumCost(echo=echo,prevCost=cost,origBHS=origBHS,origRHS=origRHS,crossedOut=crossedOut))
        
        raise RuntimeWarning("minimum Cost fell through, likely casued by an inbalanced Transport Table")
    
    def vogel(self,echo=False,origBHS=False,origRHS=False,crossedOut=[[],[]]):
        if echo:
            self.display()
        if self.RHS.sum() + self.BHS.sum() ==0:
            #all assignment done
            if len(self.BFS) + 1 == len(self.RHS) + len(self.BHS):
                return TransportTable(self.costDF.copy(),self.BFS.copy(),origRHS,origBHS)
        if type(origBHS) == bool:
            # only copy on first loop
            # otherwise pass through orig RHS and BHS
            origBHS = self.BHS.copy()
            origRHS = self.RHS.copy()
        
        RHSOppCost = Series(dtype=float)
        for i in self.RHS.index:
            rowMin = self.costDF.loc[i,:].min()
            rowMax = self.costDF.loc[i,:].max()
            rowArr = np.sort(np.unique(self.costDF.loc[i,:]))
            if (i in crossedOut[0] 
            or rowMin == rowMax):
                RHSOppCost = RHSOppCost.append(Series([0],index=[i]))
            else:
                RHSOppCost = RHSOppCost.append(Series([rowArr[1] - rowArr[0]],index=[i]))

        BHSOppCost = Series(dtype=float)
        for i in self.BHS.index:
            colMin = self.costDF.loc[:,i].min()
            colMax = self.costDF.loc[:,i].max()
            colArr = np.sort(np.unique(self.costDF.loc[:,i]))
            if (i in crossedOut[1] 
            or colMin == colMax):
                BHSOppCost = BHSOppCost.append(Series([0],index=[i]))
            else:
                BHSOppCost = BHSOppCost.append(Series([colArr[1] - colArr[0]],index=[i]))

        maxCost = RHSOppCost.append(BHSOppCost).max()
        if maxCost in RHSOppCost.array:
            for rowIndex,value in RHSOppCost.items():
                if value == maxCost and rowIndex not in crossedOut[0]:
                    toSearch = self.costDF.loc[rowIndex].copy()
                    for i in self.BHS.index:
                        if i in crossedOut[1]:
                            toSearch.loc[i] = np.inf
                    for colIndex in self.BHS.index:
                        if toSearch.loc[colIndex] == min(toSearch):
                            nextState = self.assignMax(rowIndex,colIndex)
                            rowSum = origRHS.loc[rowIndex]
                            colSum = origBHS.loc[colIndex]
                            for k,v in  nextState.BFS.items():
                                if k[0] == rowIndex:
                                    rowSum -= v
                                if k[1] == colIndex:
                                    colSum -= v
                            #only cross out one row/col
                            if (rowSum == 0 
                            and len(self.RHS) - len(crossedOut[0]) > 1):
                                crossedOut[0].append(rowIndex)
                            elif (colSum == 0                           # TODO testing if last case assigned is value 0 works
                            and len(self.BHS)-len(crossedOut[1]) > 1):  # len-len is supposed to not eliminate the last in each dir
                                crossedOut[1].append(colIndex)
                            return nextState.vogel(echo=echo,origRHS=origRHS,origBHS=origBHS,crossedOut=crossedOut)
        else: #maxCost in BHSOppCost
            for colIndex,value in BHSOppCost.items():
                if value == maxCost and colIndex not in crossedOut[1]:
                    toSearch = self.costDF.loc[:,colIndex].copy()
                    for i in self.RHS.index:
                        if i in crossedOut[0]:
                            toSearch.loc[i] = np.inf
                    for rowIndex in self.RHS.index:
                        if toSearch.loc[rowIndex] == min(toSearch):
                            nextState = self.assignMax(rowIndex,colIndex)
                            rowSum = origRHS.loc[rowIndex]
                            colSum = origBHS.loc[colIndex]
                            for k,v in  nextState.BFS.items():
                                if k[0] == rowIndex:
                                    rowSum -= v
                                if k[1] == colIndex:
                                    colSum -= v
                            #only cross out one row/col
                            if (rowSum == 0 
                            and len(self.RHS) - len(crossedOut[0]) > 1):
                                crossedOut[0].append(rowIndex)
                            elif (colSum == 0                           # TODO testing if last case assigned is value 0 works
                            and len(self.BHS)-len(crossedOut[1]) > 1):  # len-len is supposed to not eliminate the last in each dir
                                crossedOut[1].append(colIndex)
                            return nextState.vogel(echo=echo,origRHS=origRHS,origBHS=origBHS,crossedOut=crossedOut)
            
        raise RuntimeWarning("Vogel fell through, likely casued by an inbalanced Transport Table")

            # colIndex = BHSOppCost.index(maxCost)+1
            # toSearch = self.costDF.loc[:,colIndex].copy()
            # for i in self.RHS.index:
            #     if self.RHS.loc[i] == 0:
            #         toSearch.loc[i] = np.inf
            # for i in self.RHS.index:
            #     if toSearch.loc[i] == min(toSearch):
            #         return (self.assignMax(i,colIndex)
            #                     .vogel(echo=echo,origRHS=origRHS,origBHS=origBHS))

        raise RuntimeWarning("Vogel fell through, likely casued by an inbalanced Transport Table")

    # solving
    def solve(self,echo=False):
        if echo:
            self.display()
        eV = self.getEnteringVar(echo=echo)
        if eV == False: # is better to not run the calculation twice
            return self
        loop = self.findLoop((eV,),echo=echo)
        return self.loopPivot(loop,echo=echo).solve(echo=echo)
    
    def isOptimal(self):
        # reuse duality code here
        return bool(self.getEnteringVar)

    def getEnteringVar(self,echo=False):
        # SOLVE FOR U and V
        
        matrixRHS = []
        LHSIndex = Series(self.RHS.index).copy().apply(lambda x:"U" + str(x))
        LHSIndex = LHSIndex.append(Series(self.BHS.index).copy().apply(lambda x:"V" + str(x)))
        LHS_Template = Series([0]*len(LHSIndex),index=LHSIndex,dtype=int)
        matrixLHS = DataFrame(columns=LHSIndex,dtype=int)

        for k,v in self.BFS.items():
            LHSrow = LHS_Template.copy()
            LHSrow.loc["U" + str(k[0])] = 1
            LHSrow.loc["V" + str(k[1])] = 1
            matrixLHS = matrixLHS.append(LHSrow,ignore_index=True)
            matrixRHS += [self.costDF.loc[k[0],k[1]]]
        matrixRHS = Series(matrixRHS)
        matrixLHS = matrixLHS.drop(columns=["U1"])

        soln = np.linalg.solve(matrixLHS,matrixRHS)
        soln = Series(soln,index=matrixLHS.columns)
        soln = soln.append(Series([0],index=["U1"]))
        soln = soln[LHSIndex]


        basicVarVals = dict()
        w_dict = dict()
        for colIndex in self.BHS.index:
            for rowIndex in self.RHS.index:
                uVal = soln.loc[f'U{rowIndex}']
                vVal = soln.loc[f'V{colIndex}']
                cVal = self.costDF.loc[rowIndex,colIndex]

                uVal = int(uVal) if int(uVal) == uVal else uVal
                vVal = int(vVal) if int(vVal) == vVal else vVal
                cVal = int(cVal) if int(cVal) == cVal else cVal


                basicVarVals[f"U{subscript(rowIndex)}"] = uVal
                basicVarVals[f"V{subscript(colIndex)}"] = vVal

                if (rowIndex,colIndex) not in list(self.BFS.keys()):
                    if echo:
                        terms = f"U{subscript(rowIndex)} + V{subscript(colIndex)} - C{subscript(rowIndex)},{subscript(colIndex)}"
                        middle = f" = {uVal} + {vVal} - {cVal}"
                        calcValue = f" = {uVal + vVal - cVal}"
                        print(terms + middle + calcValue)
                    w_dict[rowIndex,colIndex] = soln.loc[f"U{rowIndex}"] + soln.loc[f"V{colIndex}"] - self.costDF.loc[rowIndex,colIndex]
        if echo:{print(basicVarVals)}
        maxW = max(w_dict.values())
        
        if maxW == 0:
            if echo:{print(f"all W values are <= 0, optimal solution found\nobjective value is {self.objectiveValue()}\n")}
            return False
        
        for colIndex in [i[1] for i in self.BFS.keys()]:
            for rowIndex in [i[0] for i in self.BFS.keys()]:
                if w_dict.get((rowIndex,colIndex)) == maxW:
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
            nextSteps = tuple(i for i in nextSteps if i in list(self.BFS.keys()) or i == path[0])
            out =  tuple(self.findLoop(path+(i,),axis="row",echo=echo) for i in nextSteps)
        else: #axis = "row"
            nextSteps = tuple((prevRow,i) for i in self.BHS.index)
            nextSteps = tuple(i for i in nextSteps if i not in path[1:])
            nextSteps = tuple(i for i in nextSteps if i in list(self.BFS.keys()) or i == path[0])
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
        newBFS = self.BFS.copy()
        newBFS[loopSeq[0]] = 0
        # evenIndexTuple is an array of even squares
        # oddIndexTuple is an array of odd squares
        # members are tuples of (row,col) similar to x(row,col) notation
        oddVals = [newBFS[i] for i in oddIndexTuple]
        theta = min(oddVals)
        leavingVariable = oddIndexTuple[oddVals.index(theta)]       #TODO potential degeneracy when mutiple leaving variables can exist?
        
        if echo:{print(f"Leaving Variable is: X{leavingVariable}")}

        for i in oddIndexTuple:
            newBFS[i] -= theta
        for i in evenIndexTuple:
            newBFS[i] += theta
        del newBFS[leavingVariable]
        return TransportTable(self.costDF.copy(),newBFS,self.RHS.copy(),self.BHS.copy())

    # getters
    def objectiveValue(self):
        '''
        return calculated objective value of this transport tableau using current assignments
        '''
        dispDF = DataFrame(np.zeros_like(self.costDF),index=self.costDF.index,columns=self.costDF.columns)
        for k,v in self.BFS.items():
            dispDF.loc[k[0],k[1]] = v
        dispDF.insert(dispDF.shape[1],self.RHS.name,self.RHS)
        return self.costDF.mul(dispDF).sum().sum()

    #display section
    def displayFormat(self):
        dispDF = DataFrame(np.zeros_like(self.costDF),index=self.costDF.index,columns=self.costDF.columns)
        for k,v in self.BFS.items():
            dispDF.loc[k[0],k[1]] = v
        dispDF.insert(dispDF.shape[1],self.RHS.name,self.RHS)
        dispColVars = self.BHS.copy()
        dispColVars.name = self.BHS.name
        dispDF = dispDF.append(dispColVars)
        dispDF = dispDF.applymap(lambda x:"" if np.isnan(x) else x)
        return dispDF
    
    def display(self):
        displayHelper(self.displayFormat())
    
    def __repr__(self):
        return str(self.displayFormat()) + "\n"