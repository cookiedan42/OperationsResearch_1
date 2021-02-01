from .ObjectiveFunction import ObjectiveFunction
from .Constraint import Constraint
from .NonNeg import NonNeg

# LP is
# a objective Function
# an objective funtion type
# an array of constraints


class LP():
    dualConvert = {
        "min":{
            "newObj":"max",
            "c":{">":">","=":"URS","<":"<"},
            "v":{">":"<","URS":"=","<":">"}
        },"max":{
            "newObj":"min",
            "c":{"<":">","=":"URS",">":"<"},
            "v":{">":">","URS":"=","<":"<"}
        }
    }
    def __init__(self, objectiveFunction, *constraints):
        if not isinstance(objectiveFunction, ObjectiveFunction):
            raise ValueError("invalid ObjectiveFunction")
        for i, c in enumerate(constraints):
            if not isinstance(c, Constraint):
                raise ValueError(f"invalid constraint {i} of {c}")
        self.objFunc = objectiveFunction
        self.constraints = constraints

    def standardForm(self):
        # add slack surplus variables
        # invert constraint row if RHS < 0
        # invert variable row if x1 <= 0 is a constraint
        newObjective = self.objFunc
        newConstraints = list(self.constraints)

        # non-negative RHS
        for i, c in enumerate(self.constraints):
            if c.RHS < 0:
                newConstraints[i] = c.invert()

        # non-neg constraints >= 0
        for i, c in enumerate(self.constraints):
            if isinstance(c, NonNeg) and c.sign == "<":
                # invert col
                newObjective = newObjective.invertVar(c.name)
                for j, d in enumerate(newConstraints):
                    newConstraints[j] = newConstraints[j].invertVar(c.name)
                # invert row
                newConstraints[i] = newConstraints[i].invert()
        return LP(newObjective, *newConstraints)
    
    def add_S(self):
        newConstraints = [c.add_S(i+1) for i,c in enumerate(self.constraints)]
        return LP(self.objFunc,*newConstraints)
    
    def dual(self,label="Y",labels = None):
        converter = self.dualConvert[self.objFunc.getObjType()]
        newObjective = ObjectiveFunction.new(
            converter['newObj'],
            *[i.RHS for i in self.constraints if not isinstance(i,NonNeg)],
            label='Y',labels=labels)
        # newObjective
        newVars = set()
        for c in self.constraints:
            newVars.update(c.LHS.keys())
        newVars = sorted(newVars)
        newConstraints = []
        for var in newVars:
            newConstraints.append(Constraint.new(
                *[c[var] for c in self.constraints if not isinstance(c,NonNeg)],
                converter['v'][self.nonNeg(var)],self.objFunc[var],
                label=label,labels=labels
            ))
        newConstraints+= NonNeg.fromArray(
            *[converter['c'][i.sign] for i in self.constraints if not isinstance(i,NonNeg)],
            label=label,labels=labels
        )
        return LP(newObjective,*newConstraints)

    def nonNeg(self,variable):
        temp = [c for c in self.constraints if isinstance(c,NonNeg) and c.name == variable]
        if len(temp) == 0:
            return "URS"
        elif len(temp) > 1:
            raise ValueError(f"multiple non-negative constraints for {variable} found")
        else:
            return temp[0].sign
    
    def __repr__(self):
        outString = str(self.objFunc)
        for i in self.constraints:
            outString += "\n" + str(i)
        return outString
