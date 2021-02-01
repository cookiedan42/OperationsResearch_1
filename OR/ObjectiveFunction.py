from pandas import DataFrame

from .Constraint import Constraint

from typing import List,Tuple,TypeVar
ObjectiveFunction = TypeVar("ObjectiveFunction")


class ObjectiveFunction(Constraint):
    @classmethod
    def new(cls, objType:str, *args,label:str="X", labels: List[str]=None) -> ObjectiveFunction:
        if labels == None:
            # labels overrides label
            if label.upper() in ["A", "S"]:
                raise ValueError(f"A and S are reserved variables")
            labels = [f"{label.upper()}{i+1}" for i in range(len(args))]
        elif len(labels) != len(args):
            raise ValueError(
                f"provided {len(labels)} labels for {len(args)} variables")
        else:
            label = labels[0][0]

        if objType not in ["min", "max"]:
            raise ValueError(f"{objType} is not a valid objective")

        LHS = {k: v for k, v in zip(labels, args)}
        sign = objType
        RHS = None
        return cls(label, LHS, objType, RHS)

    def __init__(self, label, LHS, objType, RHS):
        super().__init__(label, LHS, objType, RHS)

    def invertVar(self, variableName:str) -> ObjectiveFunction:
        newLHS = self.LHS.copy()
        if newLHS.get(variableName):
            newLHS[variableName] = -newLHS[variableName]
        return ObjectiveFunction(self.label, newLHS, self.sign, self.RHS)

    def getObjType(self) -> str:
        return self.sign

    def toDF(self,target:str="Z") -> DataFrame:
        if self.getObjType() == "max":
            return (
                DataFrame({f"{target}":1},index=[0])
                .join(-1*DataFrame(self.LHS,index=[0])
                .join(DataFrame({"sign":"=","RHS":0},index=[0])))
            )

        else: # objType == "min"
            return (
                DataFrame({f"-{target}":1},index=[0])
                .join(DataFrame(self.LHS,index=[0])
                .join(DataFrame({"sign":"=","RHS":0},index=[0])))
            )

        return target.join(-1*DataFrame(self.LHS,index=[0]).join(DataFrame({"sign":"=","RHS":0},index=[0])))

    def clear(self) -> ObjectiveFunction:
        return ObjectiveFunction(self.label,dict(),self.sign,None)

    def __repr__(self) -> str:
        outString = f"{self.sign}"
        factors = sorted([k for k in self.LHS.keys() if self.label in k])
        S_factors = sorted([k for k in self.LHS.keys() if "S" in k])
        A_factors = sorted([k for k in self.LHS.keys() if "A" in k])

        for k in factors+S_factors+A_factors:
            v = self.LHS[k]
            if v > 0:
                outString += f" +{v}{k}"
            elif v < 0:
                outString += f" {v}{k}"
        return outString
