import unittest
from weighted_lcs import *

class TestLCSethods(unittest.TestCase):

    def test_length(self):

        x = LCS(list('XMJYAUZ'), list('MZJAWXU'))
        print(x.lcs_weight)
        self.assertEqual(x.lcs_weight, 4)

        bck = x.backtrack_full()
        print(bck)
        self.assertEqual(''.join(bck), 'MJAU')




if __name__ == '__main__':
    unittest.main()