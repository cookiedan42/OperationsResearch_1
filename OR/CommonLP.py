from .LP import LP

class CommonLP():
    def __init__(self):
        pass

    @classmethod
    def bevco(cls):
        return LP.new(
            ["min", 2, 3],
            [[0.5, 0.25, "<", 4],
             [1, 3, ">", 20],
             [1, 1, "=", 10], ],
            [">", ">"],
            factorNames=False)

    @classmethod
    def bevcoInf(cls):
        return LP.new(
            ["min", 2, 3],
            [[0.5, 0.25, "<", 4],
             [1, 3, ">", 36],
             [1, 1, "=", 10], ],
            [">", ">"],
            factorNames=False)

    @classmethod
    def dakota(cls):
        return LP.new(
            ["max", 60, 30, 20],
            [[8, 6, 1, "<", 48],
             [4, 2, 1.5, "<", 20],
             [2, 1.5, 0.5, "<", 8],
             [0, 1, 0, "<", 5]],
            [">", ">", ">"],
            factorNames=False)

    @classmethod
    def dakotaDual(cls):
        return LP.new(
            ["min", 48, 20, 8, 5],
            [[8, 4, 2, 0, ">", 60],
             [6, 2, 1.5, 1, ">", 30],
             [1, 1.5, 0.5, 0, ">", 20], ],
            [">", ">", ">", ">"],
            factorNames=["Y₁", "Y₂", "Y₃", "Y₄"])
