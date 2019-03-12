import unittest
from weighted_lcs import *

class TestLCSethods(unittest.TestCase):

    def test_length(self):

        x = LCS(list('XMJYAUZ'), list('MZJAWXU'))
        print(x.lcs_length)
        self.assertEqual(x.lcs_length, 4)

        bck = x.backtrack(x.m-1, x.n-1)
        print(bck)
        self.assertEqual(bck, 'MJAU')




if __name__ == '__main__':
    unittest.main()