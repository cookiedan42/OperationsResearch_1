# -*- coding: utf-8 -*-

from context import OR
# from numpy import irr,arange
import unittest

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_some(self):
        costs = [
            [6,7,8],
            [15,80,78]
            ]
        col = [10,15]
        row = [15,5,5]
        t = OR.TransportTable.new(costs,col,row,colName="supply",rowName="demand")
        t2 = t.northWest(echo=True)
        print("\n----------\n")
        t3 = t.minimumCost(echo=True)
        print("\n----------\n")
        t4 = t.vogel(echo=True)
    def test_other(self):
        pass
if __name__ == '__main__':
    unittest.main()
