# -*- coding: utf-8 -*-

from context import OR
# from numpy import irr,arange
import unittest

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_BFS(self):
        costs = [
            [6,7,8],
            [15,80,78]
            ]
        RHS = [10,15]
        BHS = [15,5,5]
        # t = OR.TransportTable.new(costs,RHS,BHS,RHSName="supply",BHSName="demand")
        # t2 = t.northWest(echo=True)
        # print("\n----------\n")
        # t3 = t.minimumCost(echo=True)
        # print("\n----------\n")
        # t4 = t.vogel(echo=True)
        # print("\n----------\n")
    def test_optimal(self):
        costs = [
            [8,6,10,9],
            [9,12,13,7],
            [14,9,16,5]
            ]
        RHS = [35,50,40]
        BHS = [45,20,30,30]
        
        t = OR.TransportTable.new(costs,RHS,BHS,RHSName="supply",BHSName="demand")
        print(t)
        t2 = t.northWest(echo=False)
        print(t2)
        print("\n----------\n")
        print(t2.to_LP())
        print("\n----------\n")
        print(t2.to_DualLP())
        print("\n----------\n")
        print(t2.solve(echo=True))

if __name__ == '__main__':
    unittest.main()
