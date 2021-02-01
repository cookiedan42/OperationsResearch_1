import numpy as np
from pandas import DataFrame

from .TableauBase import TableauBase
from .LP import LP
from .NonNeg import NonNeg
from .ObjectiveFunction import ObjectiveFunction
from .Format import InfeasibleException

from typing import Dict, TypeVar
TwoPhase = TypeVar("TwoPhase")

class TwoPhase(TableauBase):

    @classmethod
    def fromLP(cls,inLP:LP) -> TwoPhase:
        inLP = inLP.standardForm()
        # origZ = super().fromLP(inLP).dataFrame.loc[0]
        # print(origZ)
        origZ = inLP.objFunc
        inLP.objFunc = ObjectiveFunction.new("min")
        for i,c in enumerate(inLP.constraints):
            if not isinstance(c,NonNeg) and c.sign in ["=",">"]:
                inLP.objFunc[f"A{i+1}"] = 1
                c[f"A{i+1}"] = 1
        baseLP = super().fromLP(inLP)
        return cls.fromBase(baseLP,origZ)

    @classmethod
    def fromBase(cls,base:TableauBase,origZ) -> TwoPhase:
        return cls(base.dataFrame,base.basicVar,origZ)

    def canonical(self,echo=False) -> TwoPhase:
        t = super().canonical(echo=echo)
        return self.fromBase(t,self.origZ)

    def __init__(self,dataFrame:DataFrame,basicVar:DataFrame,origZ:ObjectiveFunction):
        self.dataFrame = dataFrame
        self.basicVar = basicVar
        self.origZ = origZ
    
    def canonical(self,echo=False)-> TwoPhase:
        t = super().canonical(echo=echo)
        return self.fromBase(t,self.origZ)

    def phase1(self,echo=False) -> TwoPhase:
        '''
        use phase 1 to assign a BFS to the new TwoPhase Object
        '''
        self = self.canonical(echo=echo)
        newBody = super().solve(echo=echo)
        if newBody.dataFrame.loc[0,"RHS"] < 0 :
            raise InfeasibleException(f"infeasible LP\n{DataFrame(newBody.dataFrame.loc[0]).T}")
        return self.fromBase(newBody,self.origZ)

    def phase2(self,echo=False) -> TwoPhase:
        '''
        solve the tableau normally
        '''
        p2Body = self.dataFrame.copy()
        p2Body = p2Body.drop([i for i in p2Body.columns if i[0] == "A"],axis=1)
        p2Body = p2Body.drop([0],axis=0)
        newZ = self.origZ.toDF().drop("sign",axis=1)
        p2 = p2Body.append(newZ).sort_index().fillna(0)

        soln = TableauBase(p2,self.basicVar).canonical(echo=echo).solve(echo=echo)
        return self.fromBase(soln,self.origZ)

    def solve(self,echo=False):
        return self.phase1(echo=echo).phase2(echo=echo)