from pandas import DataFrame
import numpy as np

from .TableauBase import TableauBase
from .LP import LP
from .ObjectiveFunction import ObjectiveFunction
from .NonNeg import NonNeg

from typing import TypeVar
BigM = TypeVar("BigM")


class BigM(TableauBase):
    @classmethod
    def fromBase(cls,inLP:LP) -> BigM:
        return cls(inLP.dataFrame,inLP.basicVar)

    @classmethod
    def fromLP(cls,inLP:LP,M_Val: float = 1e6) -> BigM:
        inLP = inLP.standardForm()
        for i,c in enumerate(inLP.constraints):
            if not isinstance(c,NonNeg) and c.sign in ["=",">"]:
                inLP.objFunc[f"A{i+1}"] = M_Val
                c[f"A{i+1}"] = 1
        baseLP = super().fromLP(inLP)
        return cls.fromBase(baseLP)

    def __init__(self,dataFrame:DataFrame,basicVar:DataFrame,M_Val: float=1e6):
        self.dataFrame = dataFrame
        self.basicVar = basicVar
        self.M_Val = M_Val

    def canonical(self,echo=False) -> BigM:
        t = super().canonical(echo=echo)
        return self.fromBase(t)

    def isInfeasible(self):
        if not self.isOptimal():
            return False 
        elif "A" in "".join(self.getBFS().keys()):
            return False
        else:
            return True

    def solve(self,echo=False):
        t = super().canonical(echo=echo).solve(echo=echo)
        return self.fromBase(t) 