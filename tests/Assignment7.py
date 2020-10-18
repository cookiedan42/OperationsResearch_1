# -*- coding: utf-8 -*-

from context import OR
# from numpy import irr,arange
import unittest

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_some(self):
        costs = [
            [15,35,25,0],
            [10,50,40,0],
            [115,135,125,0],
            [110,150,140,0]
        ]
        RHS = [40,30,20,20]
        BHS = [30,30,30,20]
        q = OR.TransportTable.new(costs,RHS,BHS)
        r = q.northWest(echo=False)
        print(r)
        s = r.solve(echo=True)
        print(s)
    def test_other(self):
        pass
if __name__ == '__main__':
    unittest.main()