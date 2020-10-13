from pandas import DataFrame
import numpy as np
from .Format import *
from .TableauBase import TableauBase

def getRep(num,M_val):
    # solve for integer M values, the rest... soz
    round5 = lambda x: round(x,5)
    if abs(num%M_val) < abs(num%-M_val):
        #kM+x
        mNo = int((num - num%M_val)/M_val)
        defNo = num%M_val
    else:   
        #KM-x
        mNo = int((num - num%-M_val)/M_val)
        defNo = num%-M_val
    
    mNo = round5(mNo)
    defNo = round5(defNo)

    if mNo == 1:
        mString = " M "
    elif mNo == -1:
        mString = "-M "
    elif mNo == 0:
        mString = ""
    else:
        mString = f"{mNo}M "
    
    if defNo == 0:
        defString = ""
    elif defNo > 0:
        defString = f"+{defNo}"
    else:
        defString = f"{defNo}"
    if mString + defString  == "":
        return 0
    else:
        return mString + defString

class BigM(TableauBase):
    @classmethod
    def fromLP(cls,inLP,M_Val = 1e6):
        constraintSigns = inLP.posRHS().body.loc[:,"signs"]
        baseT = super().fromLP(inLP)
        body = baseT.dataFrame
        basicVar = list(baseT.basicVar)
        aVals = []
        for i in range(1,len(constraintSigns)):
            if constraintSigns[i] in (">","="):
                newCol = [0] * body.shape[0]
                newCol[0] =  M_Val
                newCol[i] = 1
                body.insert(body.shape[1]-1,f"a{subscript(i)}",newCol)
                basicVar[i] = f"a{subscript(i)}"
        
        return BigM(body,tuple(basicVar),aVals,M_Val=M_Val)
        
    def fromTableauBase(self,base):
        return BigM(base.dataFrame,base.basicVar,self.aVals)

    def __init__(self,dataFrame,basicVar,aVals,M_Val=1e6):
        super().__init__(dataFrame,basicVar,dispFunc = lambda x: getRep(x,M_Val))
        self.aVals = aVals              # list of a factors
        # self.dispFunc = 

    def canonical(self,echo=False):
        t = super().canonical(echo=echo)
        return self.fromTableauBase(t)

    def solve(self,echo=False):
        t = super().solve(echo=echo)
        return self.fromTableauBase(t) 