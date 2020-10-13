# -*- coding: utf-8 -*-

from context import OR
# from numpy import irr,arange
import unittest

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_TableauBase(self):
        dakota = OR.LP.new(
            ["max",60,30,20],
            [[8,6,1,"<",48],
            [4,2,1.5,"<",20],
            [2,1.5,0.5,"<",8],
            [0,1,0,"<",5]],
            [">",">",">"],
            factorNames = False)
        b = OR.TableauBase.fromLP(dakota)

if __name__ == '__main__':
    unittest.main()
