import tensorflow as tf
import tensorflow_hub as hub
import scipy.spatial.distance as distance

from patterns import *

print(tf.__version__)

class ElmoContext(EmbeddingContext):

    def __init__(self, spec = 'http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz') -> None:
        # self.elmo = hub.Module('http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz', trainable=False) #wiki
        # self.elmo = hub.Module('http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz', trainable=False)  # twitter
        # self.elmo = hub.Module('http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz', trainable=False) #Russian WMT News
        self.elmo = hub.Module(spec,trainable=False)

    def get_embedding_tensor(self, str):
        return self.__get_embedding_tensor([str])[0]

    def get_compare_func(self):
        def cos_sim(x, y):
            return 1-distance.cosine(x, y)
        return cos_sim

    def __get_embedding_tensor(self, l, type="elmo", signature="default"):
        # embeddings = self.elmo(
        #     inputs={
        #         "tokens": [str],
        #         "sequence_len": [len(str)]
        #     },
        #     signature="tokens",
        #     as_dict=True)["elmo"]

        embedding_tensor = self.elmo(l, signature=signature, as_dict=True)[type]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            embedding = sess.run(embedding_tensor)

        return embedding

if __name__ == '__main__':
    ec = ElmoContext()
    # sc = StringEmbeddingContext()

    ptext = 'Правительство Республики Судан и Правительство Республики Южный Судан далее называемые Cтороны принимают настоящее Соглашение'
    p = Pattern(ptext, 8, 11, embedding_context=ec)


    s = 'Государства участники настоящей Декларации именуемые в дальнейшем Cтороны будут продолжать развивать и укреплять сотрудничество в области развития железнодорожного транспорта на евроазиатском пространстве далее называемые Cтороны принимают настоящее Соглашение'
    #s = 'Государства участники настоящей Декларации далее называемые Cтороны будут продолжать развивать и укреплять сотрудничество в области развития железнодорожного транспорта на евроазиатском пространстве'
    # s = s.split()
    res = find_fuzzy_pattern(p, s)
    print('rez:', res)
    print(get_string_from_span(res, s))


    ptext2 = 'далее называемые Cтороны'
    p2 = Pattern(ptext2, 0, 3, embedding_context=ec)

    res = find_fuzzy_pattern(p2, s)
    print('rez:', res)
    print(get_string_from_span(res, s))
