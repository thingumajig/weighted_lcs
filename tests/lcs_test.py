import unittest
from weighted_lcs import *
from elmo_pattern import *
from patterns import *

class TestLCSethods(unittest.TestCase):

    def test_length(self):
        self.__do_lcs_strings('XMJYAUZ', 'MZJAWXU', 'MJAU')
        self.__do_lcs_strings('XMJYAUZ', 'MMMMMZJAWXUEEEE', 'MJAU')
        self.__do_lcs_strings('XXXXMJYAUZZZZZ', 'MZJAWXU', 'MJAU')

        self.__do_lcs_strings('XMJYAUZ', 'MZJAWXUSSSSXMJYAUZ', 'XMJYAUZ')
        self.__do_lcs_strings('XSMJAUZZZ', 'XSASSFMJAZUREDFMZZZ', 'XSMJAUZZZ')


    def __do_lcs_strings(self, s1, s2, R):
        print('\ntest s1: {} s2: {} R: {}'.format(s1, s2, R))
        x = LCS(list(s1), list(s2))
        print('lcs len:\t', x.lcs_length)

        js = ''.join(x.backtrack_list())
        print('lcs string:\t', js)

        ibck = x.backtrack_indexes()
        print('indexes:\t',ibck)
        info = x.get_full_info(ibck)
        print(info)
        (a, b) = compile_arrays(x.x, x.y, ibck)
        print('weight(max):\t', x.lcs_length / max(len(a), len(b)))
        print('weight(min):\t', x.lcs_length / min(len(a), len(b)))
        print('from s1: {}  and from s2: {}'.format(''.join(a), ''.join(b)))

        self.assertEqual(info[2], x.lcs_length / max(len(a), len(b)))
        self.assertEqual(x.lcs_length, len(R))
        self.assertEqual(js, R)

    def test_backtrack_all(self):
        self.__do_lcs_backtrack_all('AATCC', 'ACACG', '')
        self.__do_lcs_backtrack_all('XMJAUZRRRRMJAURR', 'TTMJAUEE', 'MJAU')

    def __do_lcs_backtrack_all(self, s1, s2, R):
        print('\ntest s1: {} s2: {} R: {}'.format(s1, s2, R))
        x = LCS(list(s1), list(s2))
        print('lcs len:\t', x.lcs_length)

        ibck = x.backtrack_indexes()
        print('indexes:\t',ibck)

        js = x.backtrack_all_sequences()
        print('all sequences:\t', js)

        for jss in js:
            info = x.get_full_info(jss)
            print(info)
            print('lcs: ', gather_array(x.x, jss))

            (a, b) = compile_arrays(x.x, x.y, ibck)
            print('weight(max):\t', x.lcs_length / max(len(a), len(b)))
            print('weight(min):\t', x.lcs_length / min(len(a), len(b)))
            print('from s1: {}  and from s2: {}'.format(''.join(a), ''.join(b)))

    def test_elmo(self):
        ec = ElmoContext()
        ptext = 'Правительство Республики Судан и Правительство Республики Южный Судан далее называемые Cтороны принимают настоящее Соглашение'
        p = Pattern(ptext, 8, 11, embedding_context=ec)

        self.__do_elmo_strings(p,
                               'Государства участники настоящей Декларации именуемые в дальнейшем Cтороны будут продолжать развивать и укреплять сотрудничество в области развития железнодорожного транспорта на евроазиатском пространстве',
                               'именуемые в дальнейшем Cтороны')


    def __do_elmo_strings(self, pattern, s, R):
        print('\nTest:\ns1: {}\npattern: {}\nR: {}'.format(s,pattern, R))

        res = find_fuzzy_pattern(pattern, s)
        print('rez:', res)
        rs = get_string_from_span(res, s)
        print(rs)
        self.assertEqual(R, rs)

if __name__ == '__main__':
    unittest.main()