# -*- coding: utf-8 -*-

from context import OR
# from numpy import irr,arange
import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_BigM(self):
        bevco = OR.CommonLP.bevco()
        b = OR.BigM.fromLP(bevco)
        assert type(b) == OR.BigM
        c = b.canonical(echo=False)
        assert type(c) == OR.BigM
        d = c.solve(echo=False)

    def test_infeasible(self):
        bevco = OR.CommonLP.bevcoInf()
        b = OR.BigM.fromLP(bevco)
        assert type(b) == OR.BigM
        c = b.canonical(echo= False)
        assert type(c) == OR.BigM
        d = c.solve(echo = False)

if __name__ == '__main__':
    unittest.main()