from .ObjectiveFunction import ObjectiveFunction
from .Constraint import Constraint
from .NonNeg import NonNeg
from .LP import LP

bevCo = LP(
    ObjectiveFunction.new("min", 2, 3),
    Constraint.new(0.5, 0.25, "<", 4),
    Constraint.new(1, 3, ">", 20),
    Constraint.new(1, 1, "=", 10),
    *NonNeg.fromArray(">",">",))