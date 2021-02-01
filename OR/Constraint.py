from pandas import DataFrame

from typing import List,Tuple,TypeVar
Constraint = TypeVar("Constraint")


class Constraint():
    invertSign = {">": "<", "=": "=", "<": ">"}

    @classmethod
    def new(cls, *args, label:str="X", labels=None):
        '''
        construct a new Constraint from array of values
        '''
        #label tests
        if labels == None:
            # labels overrides label
            if label.upper() in ["A", "S"]:
                raise ValueError(f"A and S are reserved variables")
            labels = [f"{label.upper()}{i+1}" for i in range(len(args)-2)]
        elif len(labels) != len(args)-2:
            raise ValueError(f"provided {len(labels)} labels for {len(args)-2} variables")
        else:
            label = labels[0][0]
        #sign tests
        if args[-2] not in ["<", ">", "="]:
            raise ValueError(f"{args[-2]} is not a valid constraint sign")

        LHS = {k: v for k, v in zip(labels, args[:-2])}
        sign = args[-2]
        RHS = args[-1]
        return cls(label, LHS, sign, RHS)

    def __init__(self, label, LHS, sign, RHS):
        self.label = label.upper()  # keep track of what variable we are using
        self.LHS = LHS
        self.sign = sign
        self.RHS = RHS
        # self.iterCount for iterable

    def sign(self) -> str:
        return self.sign

    def RHS(self) -> float:
        '''
        all numbera are subtypes of float i think><
        '''
        return self.RHS

    def __getitem__(self, key:str) -> float:
        if self.LHS.get(key):
            return self.LHS.get(key)
        else:
            return 0

    def __setitem__(self, key:str, value: float):
        self.LHS[key] = value

    def clear(self) -> Constraint:
        return Constraint(self.label,dict(),self.sign,0)

    def add_S(self, sIndex: int) -> Constraint:
        newLHS = self.LHS.copy()
        if self.sign == ">":
            newLHS[f"S{sIndex}"] = -1
        elif self.sign == "<":
            newLHS[f"S{sIndex}"] = 1
        return Constraint(self.label,newLHS,'=',self.RHS)

    def invert(self) -> Constraint:
        return Constraint(
            self.label,
            {k:-v for k,v in self.LHS.items()},
            self.invertSign[self.sign],
            -self.RHS)

    def invertVar(self, variableName:str) -> Constraint:
        newLHS = self.LHS.copy()
        if newLHS.get(variableName):
            newLHS[variableName] = -newLHS[variableName]
        return Constraint(
            self.label,
            newLHS,
            self.sign,
            self.RHS)

    def addVar(self,newVars: List[str]) -> Constraint:
        newLHS = self.LHS.copy()
        for i in newVars:
            newLHS[i] = 0
        return Constraint(
            self.label,
            newLHS,
            self.sign,
            self.RHS)

    def toDF(self,index=0)->DataFrame:
        return DataFrame(self.LHS,index=[index]).join(DataFrame({"sign":self.sign,"RHS":self.RHS},index=[index]))

    def get_variables(self) -> List[str]:
        return list(self.LHS.keys())

    def __len__(self) -> int:
        return len(self.LHS)

    def __repr__(self):
        outString = ""
        factors = sorted([k for k in self.LHS.keys() if self.label in k])
        S_factors = sorted([k for k in self.LHS.keys() if "S" in k])
        A_factors = sorted([k for k in self.LHS.keys() if "A" in k])

        for k in factors+S_factors+A_factors:
            v = self.LHS[k]
            if v > 0:
                outString += f" +{v}{k}"
            elif v < 0:
                outString += f" {v}{k}"
        outString += f" {self.sign} {self.RHS}"
        return outString[1:]