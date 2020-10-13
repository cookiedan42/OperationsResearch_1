# -*- coding: utf-8 -*-

from context import OR
# from numpy import irr,arange
import unittest
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_subscript(self):
        assert OR.subscript(1) == "₁"
        assert OR.subscript(12) == "₁₂" 
        
    def test_CommonLP(self):
        assert OR.CommonLP.bevco() == OR.LP.new(
            ["min", 2, 3],
            [[0.5, 0.25, "<", 4],
             [1, 3, ">", 20],
             [1, 1, "=", 10], ],
            [">", ">"],
            factorNames=False)

if __name__ == '__main__':
    unittest.main()
