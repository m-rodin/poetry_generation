from gensim.models.keyedvectors import KeyedVectors


class GloveEmbedder:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format('./glove/glove.6B.300d.word2vec.txt', binary=False)
        
    def exist_emb(self, word):
        return word in self.model.vocab.keys()
    
    def get_emb(self, word):
        return self.model[word]
