import tensorflow as tf
import tensorflow_hub as hub
import scipy.spatial.distance as distance

from patterns import *

print(tf.__version__)

class ElmoContext(EmbeddingContext):

    def __init__(self) -> None:
        # self.elmo = hub.Module('http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz', trainable=False) #wiki
        self.elmo = hub.Module(
            'http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz',
            trainable=False)  # twitter
        # self.elmo = hub.Module('http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz', trainable=False) #Russian WMT News

    def get_embedding_tensor(self, list):
        return self.get_embedding_tensor(list)

    def get_compare_func(self):
        return distance.cosine

    def get_embedding_tensor(self, str, type="elmo", signature="default"):
        embedding_tensor = self.elmo(str, signature=signature, as_dict=True)[type]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            embedding = sess.run(embedding_tensor)

        return embedding

if __name__ == '__main__':
    ec = ElmoContext()
    ptext = 'Правительство Республики Судан и Правительство Республики Южный Судан далее называемые Cтороны принимают настоящее Соглашение.'
    p = Pattern(ptext, 8, 11, embedding_context=ec)
    s = 'Государства участники настоящей Декларации именуемые в дальнейшем Стороны будут продолжать развивать и укреплять сотрудничество в области развития железнодорожного транспорта на евроазиатском пространстве'
    s = s.split()
    res = find_fuzzy_pattern(p, s)
    print('rez:', res)
    print(' '.join(s[res[1][0]:res[1][1]]))


