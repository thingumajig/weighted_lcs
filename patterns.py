from typing import Any, Tuple, Union, List

from weighted_lcs import LCS
import numpy as np

class EmbeddingContext:

    def get_embedding_tensor(self, str):
        pass

    def get_compare_func(self):
        pass


class CharEmbeddingContext(EmbeddingContext):
    def get_embedding_tensor(self, str):
        return np.asarray(list(str))   #for compatibility

    def get_compare_func(self):
        def simple_compare(x, y):
            return 1. if x == y else 0.
        return simple_compare


class SimpleTokenEmbeddingContext(CharEmbeddingContext):
    def get_embedding_tensor(self, str):
        return np.asarray(str.split()) #for compatibility


class Pattern:
    def __init__(self, text, start, stop, embedding_context: EmbeddingContext = CharEmbeddingContext()):
        self.start = start
        self.stop = stop
        self.embedding_context = embedding_context
        self.embedding = self.embedding_context.get_embedding_tensor(text)
        print('Pattern shape: {}'.format(self.get_pattern_embedding().shape))

    def get_pattern_embedding(self):
        return np.asarray(self.embedding[self.start:self.stop])

    def get_matcher(self, text_embedding_list: list):
        return Matcher(self, text_embedding_list)

    def get_matcher_str(self, text: str):
        text_emb_list: list = self.embedding_context.get_embedding_tensor(text)
        return self.get_matcher(text_emb_list)

    def get_pattern_len(self):
        return self.stop - self.start

class Matcher:

    def __init__(self, pattern: Pattern, text_embedding_list: list) -> None:
        super().__init__()
        self.pattern = pattern
        self.text_emb = text_embedding_list
        self.lcs = LCS(text_embedding_list, pattern.get_pattern_embedding(), compare = pattern.embedding_context.get_compare_func())

        self.i, self.j = self.lcs.m-1, self.lcs.n-1

    def find(self):
        indexes = self.lcs.backtrack_indexes(self.i, self.j)
        print(indexes)
        print('lcs len:', self.lcs.lcs_length)
        if indexes:
            s1, s2, w = self.lcs.get_full_info(indexes)

            weight = w / max(self.pattern.get_pattern_len(), s1[1] - s1[0], s2[1] - s2[0])
            if weight < self.lcs.threshold:
                return None, None

            self.i, self.j = s1[0], self.lcs.n-1
            self.lcs.lcs_length = self.lcs.matrix[self.i, self.j]

            return weight, s1
        else:
            return None, None


def find_fuzzy_pattern(pattern: Pattern, text: list) -> Tuple[float, Tuple[int, int]]:
    text_emb_list = pattern.embedding_context.get_embedding_tensor(text)
    return find_fuzzy_pattern_emb(pattern, text_emb_list)


def find_fuzzy_pattern_emb(pattern, text_emb_list) -> Tuple[float, Tuple[int, int]]:
    lcs = LCS(text_emb_list, pattern.get_pattern_embedding(),
              compare=pattern.embedding_context.get_compare_func())

    span1, span2, weight = lcs.backtrack_full()
    # if weight > lcs.threshold:
    return weight, span1


def find_all_patterns(pattern, text_emb_list) -> list:
    lcs = LCS(text_emb_list, pattern.get_pattern_embedding(),
              compare=pattern.embedding_context.get_compare_func())

    i, j = lcs.m-1, lcs.n-1

    spans = []
    while i > 0 and j > 0:
        indexes = lcs.backtrack_indexes(i,  j)
        print(indexes)
        s1, s2, w = lcs.get_full_info(indexes)
        spans.append((s1, w))
        i = s1[0]

    return spans

def get_string_from_tuple(res, s, delimiter=' '):
    w, span = res
    return get_string_from_span(span, s, delimiter=delimiter)


def get_string_from_span(span, s, delimiter=' '):
    return delimiter.join(s[span[0]:span[1]])


if __name__ == '__main__':
    ec = CharEmbeddingContext()
    p = Pattern('XSMJAUZZZ', 2, 6, embedding_context=ec)
    s = 'XSASSFMJAZUREDFMZZZMJAUZPPP'
    res = find_fuzzy_pattern(p, list(s))
    print('rez:', res)
    print(get_string_from_tuple(res, s))

    m = p.get_matcher_str(s)
    while True:
        w, span = m.find()
        if span is None:
            break

        print('span:{} w: {}'.format(span, w))
        print(get_string_from_span(span, list(s), delimiter=''))


