import unittest
from weighted_lcs import *

class TestLCSethods(unittest.TestCase):

    def test_length(self):
        self.test_lcs_strings('XMJYAUZ', 'MZJAWXU', 'MJAU')
        self.test_lcs_strings('XMJYAUZ', 'MMMMMZJAWXUEEEE', 'MJAU')
        self.test_lcs_strings('XXXXMJYAUZZZZZ', 'MZJAWXU', 'MJAU')

        self.test_lcs_strings('XMJYAUZ', 'MZJAWXUSSSSXMJYAUZ', 'XMJYAUZ')


    def test_lcs_strings(self, s1, s2, R):
        print('\ntest s1: {} s2: {} R: {}'.format(s1, s2, R))
        x = LCS(list(s1), list(s2))
        print('lcs len:\t', x.lcs_length)

        js = ''.join(x.backtrack_list())
        print('lcs string:\t', js)

        ibck = x.backtrack_indexes()
        print('indexes:\t',ibck)
        (a, b) = LCS.compile_arrays(x.x, x.y, ibck)
        print('weight(max):\t', x.lcs_length / max(len(a), len(b)))
        print('weight(min):\t', x.lcs_length / min(len(a), len(b)))
        print('from s1: {}  and from s2: {}'.format(''.join(a), ''.join(b)))

        self.assertEqual(x.lcs_length, len(R))
        self.assertEqual(js, R)



if __name__ == '__main__':
    unittest.main()