# -*- coding: utf-8 -*-

from context import OR
# from numpy import irr,arange
import unittest

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_Q1(self):
        costs = [
            [15,35,25,0],
            [10,50,40,0],
            [115,135,125,0],
            [110,150,140,0]
        ]
        RHS = [40,30,20,20]
        BHS = [30,30,30,20]
        q = OR.TransportTable2.new(costs,RHS,BHS)
        print(q)
        r = q.northWest(echo=False)
        print(r)
        s = r.solve(echo=True)
        print(s)

    def test_Q2(self):
        costs = [
            [5,4,2,0],
            [3,4,5,0]
        ]
        RHS = [10000,6000]
        BHS = [5000,5000,5000,1000]
        q = OR.TransportTable2.new(costs,RHS,BHS)
        r = q.minimumCost(echo=False)
        print(r)
        s = r.solve(echo=True)
        print(s)

    def test_Q3(self):
        costs = [
            [10,11,18,0,0],
            [ 6, 7, 7,0,0],
            [ 7, 8, 5,0,0],
            [ 5, 6, 4,0,0],
            [ 9, 4, 7,0,0]
        ]
        h = OR.Hungarian.new(costs)
        j = h.solve(echo=True)
        print(j)
        print(j.objectiveValue())


if __name__ == '__main__':
    unittest.main()