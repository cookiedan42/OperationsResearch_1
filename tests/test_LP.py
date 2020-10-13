# -*- coding: utf-8 -*-

from context import OR
# from numpy import irr
import unittest

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_dual(self):
        a = OR.LP.new(
            ["max",60,30,20],
            [[8,6,1,"<",48],
            [4,2,1.5,"<",20],
            [2,1.5,0.5,"<",8],
            [0,1,0,"<",5]],
            [">",">",">"],
            factorNames = False)
        aDual = a.getDual()
        b = OR.LP.new(
            ["min",48,20,8,5],
            [[8,4,2,0,">",60],
            [6,2,1.5,1,">",30],
            [1,1.5,0.5,0,">",20],
            ],
            [">",">",">",">"],
            factorNames = ["Y₁","Y₂","Y₃","Y₄"])
        assert aDual == b
if __name__ == '__main__':
    unittest.main()
