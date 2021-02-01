from .Constraint import Constraint

class NonNeg(Constraint):
    @classmethod
    def fromArray(cls, *args, label="X", labels=None):
        outList = []
        if labels == None:
            labels = [f"{label}{i+1}" for i in range(len(args))]
        for i, v in enumerate(args):
            if v.upper() != "URS":
                outList += [cls(labels[i], v)]
        return outList

    def __init__(self, variable, sign, RHS=0):
        self.name = variable
        self.label = variable[0]
        self.LHS = {variable: 1}
        self.sign = sign
        self.RHS = RHS

    def invertVar(self, variableName):
        if variableName == self.name:
            return NonNeg(self.name,self.sign, RHS=self.RHS)
        else:
            return self

    def add_S(self,index):
        return self

    def invert(self):
        return NonNeg(self.name, self.invertSign[self.sign], RHS=-self.RHS)
