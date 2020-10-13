from pandas import DataFrame
import pandas as pd
import numpy as np
from .Format import *

'''
keep it nice and simple 
just solve a tableau 
until row 0 has no more negative numbers
or we face some other issue
'''

def round5(x):
    # default dispFunc
    return round(x,5)

class TableauBase():
    def __init__(self,dataFrame,basicVar,dispFunc):
        self.dataFrame = dataFrame
        self.basicVar = tuple(basicVar)
        self.dispFunc = dispFunc

    def setBFS(self,bfs):
        # takes in BFS in sequence down the tableau
        out = self
        for i in range(len(bfs)):
            out = out.pivot(bfs[i],i+1)
        return out

    def canonical(self,echo=False):
        if echo:
            print("pivoting on all basic Vars")
        for var in self.basicVar[1:]:
            self = self.pivot(var,self.pickPivotRow(var,echo=echo))
            if echo:{print(f"making {var} canonical\n")}
        if echo:
            self.display()
            print("In canonical form\n")
        return self

    #pivot action
    def autoPivot(self,echo=False):
        '''
        stack a search and a pivot together, single cycle
        '''
        #echo is to toggle printing of steps
        # set other non-basic factors to 0
        # solve for row maximum value of interested factor where basic variables are 0 to inf
        t = self.dataFrame
        factor,row = self.pickPivot(echo)
        if row == None:
            print("unbounded lp")
            return self
        return self.pivot(factor,row)
    def pivot(self, factor, row):
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

        newVar = self.basicVar[:row]+(factor,)+self.basicVar[row+1:]
        return TableauBase(result,newVar,self.dispFunc)
    def pickPivot(self,echo=False):
        '''
        combining pickRow and pickCol
        '''
        t = self.dataFrame
        factor = self.pickPivotCol(echo)
        # if factor == None:      # unbounded
        #     return None,None
        row = self.pickPivotRow(factor,echo)
        if row == None:         # unbounded
            return None, None
        if echo:
            print(f"pivoting on {factor} and row {row}\n")
        return factor,row
    def pickPivotCol(self,echo=False):
        '''
        get the most negative column
        '''
        t = self.dataFrame
        t2 = t.iloc[:,:-1].iloc[0]          # objective function
        minT = min(t2)                      # value of most negative factor
        factor = None
        for i in t.columns:                
            if t[i][0] == minT:             # get entering variable with the lowest index
                factor = i                  # aka closest to the left of tableau
                break                       # Bland's rule
        return factor
    def pickPivotRow(self,factor,echo=False):
        '''
        ratio test
        1) solution divide entering col must be positive
        2) eliminate all negatives and infs
        3) take smallest of all remaining positive non inf
        4) --> such that pivoting on this will cause next solution set to remain positive no.
        '''
        t = self.dataFrame
        # ratio = t.iloc[1:,-1]/t.loc[1:,factor]
        top = t.iloc[1:,-1]
        bot = t.loc[1:,factor]
        ratio = top.copy()

        for i in range(len(ratio)):
            if bot.iloc[i] == 0:
                ratio.iloc[i] = np.inf
            else:
                ratio.iloc[i] = top.iloc[i]/bot.iloc[i]

        for i in range(len(ratio)):
            if ratio.iloc[i] < 0:           # replace if opposite sign --> values can go to inf
                ratio.iloc[i] = np.inf      # replace -inf from x/-0.0
            elif ratio.iloc[i] == -np.inf:
                ratio.iloc[i] = np.inf
        row = pd.to_numeric(ratio).idxmin() #returns first index of smallest value, follows bland's rule
        
        enterCol = np.array([""]*len(self.basicVar))
        if ratio[row] == np.inf:
            row = None
        else:
            enterCol[row] = "*"
        if echo:
            t = t.applymap(self.dispFunc)
            displayHelper(
                t.assign(basicVar=self.basicVar).join(ratio.to_frame(name="ratioTest")).assign(Entering=enterCol)
            )
        return row

    @classmethod #constructor from LP
    def fromLP(cls,inLP):
        LP2 = inLP.getStandardForm()
        body = LP2.body
        body = body.drop("signs",axis=1)
        if LP2.objectiveType == "min":
            target = "-Z"
        else:
            target = "Z"
            body.iloc[0] = body.iloc[0]*-1
        body.insert(0,target,[1]+[0]*(body.shape[0]-1))
        body.iloc[0,-1] = 0

        basicVars  = ["Z"] + [f"s{subscript(i)}" for i in range(1,body.shape[0])]
        return TableauBase(body,basicVars,round5)

    #displays/getters
    def display(self):
        return displayHelper(self.displayfmt())
    def displayfmt(self): #default round to 5 decimal places for floating point sanity
        out = self.dataFrame
        out = out.applymap(self.dispFunc)
        df =  self.dataFrame.assign(Basic = self.basicVar)
        return out.assign(Basic = self.basicVar).loc[:, (df != 0).any(axis=0)]
    def __repr__(self):
        return str(self.displayfmt())
    def getBFS(self):
        allVar = self.dataFrame.columns
        bfs = "{"
        for var in allVar[:-1]:
            value = 0
            if var in self.basicVar:
                value = self.dataFrame.iloc[self.basicVar.index(var),-1]
                value = self.dispFunc(value)

            if  var[0] =="-":
                var = var[1:]
                value*=-1



            bfs += f" {var} = {value},"
        bfs = bfs[:-1]
        bfs += " }"
        return bfs
    #comparisons
    def equals(self,other):
        diff = self.dataFrame-other.dataFrame
        return diff.apply(lambda x: abs(x)<1e-15).all().all()
    
    def solve(self,echo=False,past=[]):
        if self in past: 
            #looped
            if echo:
                self.display()
                print("Solved")
            return self
        elif all( [i>=0 for i in self.dataFrame.iloc[0,:-1]] ):
            # no more negative row 0 factors
            if echo:
                self.display()
                print("Solved")
            return self
        else:
            past +=[self]
            t1 = self.autoPivot(echo=echo)
            return t1.solve(echo=echo,past=past)

    # graveyard
    @classmethod
    def genericTableau(cls,objective,constraints,factorNames=None):
        factorNo = len(objective) - 1
        if not all([len(i)-2 == factorNo for i in constraints]):
            raise ValueError("inconsistent number of factors")
        elif objective[0] not in ["max","min"]:
            raise ValueError("invalid objective function min or max")
        elif not all([i[-2] in ["<","=",">"] for i in constraints]):
            raise ValueError("constraints must have =, < or >")
        elif factorNames: #validate col if it exists
            if not(
                (type(factorNames)==list) and 
                (len(factorNames)==factorNo) and 
                (all([type(i)==str for i in factorNames]))
                ):
                raise ValueError("invalid factorNames")
        
        if factorNames == None:
            factorNames = [f"X{subscript(i+1)}" for i in range(len(factorNo))]

        #basic factors
        factors = np.array([np.array(objective[1:])]+[i[:-2] for i in constraints])
        if objective[0] == "max":
            factors[0] *=-1
        #non-basic variables
        nonBasic = np.array([[0] + [0 for i in constraints]]).T       
        solution = np.array([[0] + [i[-1] for i in constraints]]).T
        if objective[0] == "min":
            target = "-z"
        else:
            target = "z"    
        
        header = [target] + factorNames + ["Solution"]
        tabData = np.hstack((nonBasic,factors,solution))
        basicVars = (target,)+tuple(factorNames)
        return header,tabData,basicVars
