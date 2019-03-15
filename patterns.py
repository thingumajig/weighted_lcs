from weighted_lcs import LCS

class EmbeddingContext:

    def get_embedding_tensor(self, list):
        pass

    def get_compare_func(self):
        pass


class StringEmbeddingContext(EmbeddingContext):
    def get_embedding_tensor(self, list):
        return list

    def get_compare_func(self):
        def simple_compare(x, y):
            return 1. if x == y else 0.
        return simple_compare


class Pattern:
    def __init__(self, text, start, stop, embedding_context = StringEmbeddingContext()):
        if isinstance(text, list):
            self.list = text
        else:
            self.list = text.split()

        self.start = start
        self.stop = stop
        self.embedding_context = embedding_context
        print('Pattern: {}'.format(self.get_pattern_rep()))

    def get_pattern_rep(self):
        return self.list[self.start:self.stop]

    def get_embedding(self):
        return self.embedding_context.get_embedding_tensor(self.list)

    def get_pattern_embedding(self):
        return self.get_embedding()[self.start:self.stop]


def find_fuzzy_pattern(pattern: Pattern, text: list):
    text_emb_list = pattern.embedding_context.get_embedding_tensor(text)
    lcs = LCS(text_emb_list, pattern.get_pattern_embedding(),
              compare = pattern.embedding_context.get_compare_func())

    span1, span2, weight = lcs.backtrack_full()

    if weight > lcs.threshold:
        return weight, span1


if __name__ == '__main__':
    ec = StringEmbeddingContext()
    p = Pattern(list('XSMJAUZZZ'), 2, 6, embedding_context=ec)
    s = 'XSASSFMJAZUREDFMZZZ'
    res = find_fuzzy_pattern(p, list(s))
    print('rez:', res)
    print(''.join(list(s)[res[1][0]:res[1][1]]))

