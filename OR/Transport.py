from .Format import *
from .LP import LP
import numpy as np
from pandas import DataFrame,Series


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

    def objectiveValue(self):
        '''
        return calculated objective value of this transport tableau using current assignments
        '''
        return self.costDF.mul(self.assignDF).sum().sum()

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
                rowIndex = 1+ self.RHS.to_list().index(i)
                break
        for BHSVal in self.BHS:
            if BHSVal!=0:
                colIndex = 1+self.BHS.to_list().index(i)
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
                        and self.BHS.loc[colIndex] != 0
                    ):
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

    def getEnteringVar(self):
        #think about reindexing dualBody by Xij factors
        dual = self.to_DualLP()
        dualBody = dual.body
        # dualBody.index = self.assignDF.columns
        dualBody = dualBody.drop([0])
        dualBody.index = self.to_LP().body.columns[:-2]

        counter=0
        toDrop = []
        for rowIndex in self.assignDF.index:
            for colIndex in self.assignDF.columns:
                if self.assignDF.loc[rowIndex,colIndex] == 0:
                    toDrop.append(dualBody.index[counter])
                counter+=1

        dropNonBasic = dualBody.drop(toDrop)
        #convert all contents to float
        LHS = dropNonBasic.drop(["signs","RHS"],axis=1).applymap(lambda x:float(x))
        RHS  = dropNonBasic.loc[:,"RHS"].apply(lambda x:float(x))
        # if not square
        droppedColIndex=[]
        while LHS.shape[0] != LHS.shape[1]:
            if LHS.shape[0] > LHS.shape[1]:
                LHS.insert(0,"dummy",[0]*LHS.shape[1])  #TODO test add col method
            else: # LHS.shape[0] < LHS.shape[1]
                for i in range(LHS.shape[1]):
                    if LHS.iloc[:,i].sum() == 1:
                        LHS = LHS.drop(columns=[LHS.columns[i]])
                        droppedColIndex.append(i)
                        break
        soln = np.linalg.solve(LHS,RHS)
        for val in range(len(droppedColIndex)-1,-1,-1):
            soln = np.insert(soln,droppedColIndex[val],0)

        diffs = np.subtract(
                dualBody.drop(columns=["signs","RHS"]).applymap(lambda x:float(x)).dot(soln),
                dualBody.loc[:,"RHS"].apply(lambda x:float(x))
            )
        enteringVal = max(diffs)
        enteringIndex = list(diffs).index(enteringVal)
        enteringVar = dualBody.index[enteringIndex]


        #then like idk how to do the looping >:(


        return enteringVar

    def loopPivot(self,even,odd):
        '''
        given that we have somehow found the even and odd squares,
        now we run the loopPivot on the Transport Table
        '''

        #even is an array of even squares
        # odd is an array of odd squares
        return self


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