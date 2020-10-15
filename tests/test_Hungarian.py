# -*- coding: utf-8 -*-

from pandas.core.frame import DataFrame
from context import OR
# from numpy import irr,arange
import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_Hungarian(self):
        lectEx = [
            [14, 5, 8, 7],
            [2, 12, 6, 5],
            [7, 8, 3, 9],
            [2, 4, 6, 10]
        ]

        assign = DataFrame([
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0],
            [1,0,0,0],
        ])
        assign.columns = assign.columns + 1
        assign.index = assign.index + 1
        reducedDF = DataFrame([
            [10,0,3,0],
            [0,9,3,0],
            [5,5,0,4],
            [0,1,3,5],
        ]).applymap(lambda x:float(x))
        reducedDF.columns = reducedDF.columns + 1
        reducedDF.index = reducedDF.index + 1

        h = OR.Hungarian.new(lectEx)
        j = h.solve(echo=False)

        assert j.reducedDF.equals(reducedDF)
        assert j.assignedDF.equals(assign)
if __name__ == '__main__':
    unittest.main()
