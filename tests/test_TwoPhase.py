# -*- coding: utf-8 -*-

from context import OR
# from numpy import irr,arange
import unittest

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_some(self):
        tp = OR.TwoPhase.fromLP(OR.CommonLP.bevco())
        tp = tp.solve(echo=False)

if __name__ == '__main__':
    unittest.main()
