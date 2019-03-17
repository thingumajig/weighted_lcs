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


def find_fuzzy_pattern(pattern: Pattern, text: list):
    text_emb_list = pattern.embedding_context.get_embedding_tensor(text)
    lcs = LCS(text_emb_list, pattern.get_pattern_embedding(),
              compare=pattern.embedding_context.get_compare_func())

    span1, span2, weight = lcs.backtrack_full()

    # if weight > lcs.threshold:
    return weight, span1


if __name__ == '__main__':
    ec = CharEmbeddingContext()
    p = Pattern('XSMJAUZZZ', 2, 6, embedding_context=ec)
    s = 'XSASSFMJAZUREDFMZZZ'
    res = find_fuzzy_pattern(p, list(s))
    print('rez:', res)
    print(''.join(list(s)[res[1][0]:res[1][1]]))

