from typing import Any, Tuple, Union, List

from weighted_lcs import LCS, Weightable, SimpleWeight
import numpy as np

import logging
logger = logging.getLogger(__name__)


class EmbeddingContext:
    def get_embedding_tensor(self, str):
        pass

    def get_embedding_tensors(self, l: list):
        return [self.get_embedding_tensor(x) for x in l]

    def get_compare_func(self):
        pass


class CharEmbeddingContext(EmbeddingContext):
    def get_embedding_tensor(self, str):
        return np.asarray(list(str))   #for compatibility

    def get_compare_func(self):
        def simple_compare(x, y):
            return SimpleWeight(1.) if x == y else SimpleWeight(0.)
        return simple_compare


class SimpleTokenEmbeddingContext(CharEmbeddingContext):
    def get_embedding_tensor(self, str):
        return np.asarray(str.split()) #for compatibility


class Pattern:
    def __init__(self, text, start, stop, embedding_context: EmbeddingContext = CharEmbeddingContext(), var_positions=[]):
        self.start = start
        self.stop = stop
        self.var_positions = var_positions
        self.embedding_context = embedding_context
        self.embedding = self.embedding_context.get_embedding_tensor(text)
        logger.info(f"Pattern shape: {self.get_pattern_embedding().shape}")

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
        logger.info(indexes)
        logger.info(f'lcs len: {self.lcs.lcs_length}')
        if indexes:
            # s1, s2, w = self.lcs.get_full_info(indexes)
            (i1, j1, w1) = indexes[0]
            (i2, j2, w2) = indexes[-1]
            s1, s2 = (i1, i2 + 1), (j1, j2 + 1)
            sw = 0.
            for p1, p2, wi in indexes:
                sw += wi.get_weight()

            weight = sw / self.pattern.get_pattern_len()
            logger.info(f'weight = {weight}')
            if weight < self.lcs.threshold:
                return None, None

            self.i, self.j = s1[0], self.lcs.n-1
            self.lcs.lcs_length = self.lcs.matrix[self.i, self.j]

            return weight, s1
        else:
            return None, None


class Match:
    def __init__(self, span, w) -> None:
        super().__init__()
        self.span = span
        self.weight = w


def find_fuzzy_pattern(pattern: Pattern, text: list) -> Tuple[float, Tuple[int, int]]:
    text_emb_list = pattern.embedding_context.get_embedding_tensor(text)
    return find_fuzzy_pattern_emb(pattern, text_emb_list)


def find_fuzzy_pattern_emb(pattern, text_emb_list) -> Tuple[float, Tuple[int, int]]:
    lcs = LCS(text_emb_list, pattern.get_pattern_embedding(),
              compare=pattern.embedding_context.get_compare_func())

    span1, span2, weight = lcs.backtrack_full()
    # if weight > lcs.threshold:
    return weight, span1


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
    logger.info('rez:', res)
    logger.info(get_string_from_tuple(res, s))

    m = p.get_matcher_str(s)
    while True:
        w, span = m.find()
        if span is None:
            break

        logger.info(f'span:{span} w: {w}')
        logger.info(get_string_from_span(span, list(s), delimiter=''))


