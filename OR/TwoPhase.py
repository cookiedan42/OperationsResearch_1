from .TableauBase import TableauBase
from .Format import *
import numpy as np

class TwoPhase(TableauBase):
    @classmethod
    def fromLP(cls,inLP):
        constraintSigns = inLP.posRHS().body.loc[:,"signs"]
        baseT = super().fromLP(inLP)
        body = baseT.dataFrame
        origZ = body.loc[0]
        newW = np.zeros_like(origZ)
        newW[0] = 1
        body.loc[0] = newW
        basicVar = list(baseT.basicVar)
        basicVar[0] = "-W"
        body.columns = ["-W"] + list(body.columns)[1:]
        aVals = []
        
        for i in range(1,len(constraintSigns)):
            if constraintSigns[i] in (">","="):
                newCol = [0] * body.shape[0]
                newCol[0] =  1
                newCol[i] = 1
                body.insert(body.shape[1]-1,f"a{subscript(i)}",newCol)
                basicVar[i] = f"a{subscript(i)}"
                aVals.append(f"a{subscript(i)}")
        
        #LP to phase 1
        return TwoPhase(body,tuple(basicVar),origZ,tuple(aVals))

    def fromTableauBase(self,base):
        return TwoPhase(base.dataFrame,base.basicVar,self.origZ,self.aVals)

    def __init__(self,dataFrame,basicVar,origZ,aVals):#,origObjRow):#phase1z,phase2z):
        super().__init__(dataFrame,basicVar,lambda x:round(x,5))
        self.origZ = origZ
        self.aVals = aVals

    def canonical(self,echo=False):
        t = super().canonical(echo=echo)
        return self.fromTableauBase(t)

    def solve(self,echo=False):
        # create Phase1
            # done by constructor
        #solve Phase1
        t = super().canonical(echo=echo)
        t = t.solve(echo=echo)
        # t = t.rawSolve(echo=echo)
        if t.dataFrame.iloc[0,-1]>1e-15:
            #infeasible
            print("Infeasible")
            return t
        #create Phase2
        t2 = t.dataFrame.copy()
        for a in self.aVals:
            t2 = t2.drop(a,axis=1)
        t2.columns = self.origZ.index
        t2.iloc[0] = self.origZ
        t2 = TableauBase(t2,(self.origZ.index[0],)+t.basicVar[1:],self.dispFunc)
        #solve Phase2
        t2 = t2.canonical(echo=echo)
        t2 = t2.solve(echo=echo)
        return self.fromTableauBase(t2)