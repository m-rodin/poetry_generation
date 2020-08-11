# +
from nltk import tokenize
from sentence_transformers import SentenceTransformer
from gensim.models.keyedvectors import KeyedVectors
from src.source_prepocess import simple_en_clean

from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors

import os
import numpy as np


# -

def cos_sim(w1, w2):
    return np.dot(w1, w2) / np.linalg.norm(w1) / np.linalg.norm(w2)


class FasttextBinEmbedder:
    def __init__(self, file, **kwargs):
        file_path = datapath(os.path.abspath(file))
        self.model = load_facebook_vectors(file_path)
        self.embs_cache = {}
        
    def exist_emb(self, word):
        return True
        #return word in self.model.vocab
    
    def _get_topic_emb(self, topic):
        if topic not in self.embs_cache:
            self.embs_cache[topic] = np.mean([self.model[word] for word in topic.split()], axis=0)

        return self.embs_cache[topic]
    
    def get_dist(self, topic, word):
        return cos_sim(
            self._get_topic_emb(topic),
            self.model[word]
        )


class GloveEmbedder:
    def __init__(self, file, **kwargs):
        self.model = KeyedVectors.load_word2vec_format(file, binary=False)
        self.embs_cache = {}
        
    def exist_emb(self, word):
        return word in self.model.vocab.keys()
    
    def _get_topic_emb(self, topic):
        if topic not in self.embs_cache:
            self.embs_cache[topic] = np.mean([self.model[word] for word in topic.split()], axis=0)

        return self.embs_cache[topic]
    
    def get_dist(self, topic, word):
        return cos_sim(
            self._get_topic_emb(topic),
            self.model[word]
        )


class SentBertEmbedder:
    def __init__(self, texts, **kwargs):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        
        sentences = tokenize.sent_tokenize(texts)
        sentence_embeddings = self.model.encode(sentences)
        
        clean_sents = [" ".join([simple_en_clean(w) for w in sent.split()]) for sent in sentences]
        
        self.sent2emb = dict(zip(clean_sents, sentence_embeddings))
        self.embs_cache = {}
        
    def _get_topic_emb(self, topic):
        if topic not in self.embs_cache:
            self.embs_cache[topic] = self.model.encode([topic])[0]
        return self.embs_cache[topic]
        
    def exist_emb(self, word):
        return True
    
    def get_dist(self, topic, word):
        topic_emb = self._get_topic_emb(topic)
        
        dists = []
        for sent in self.sent2emb.keys():
            if word in sent.lower():
                dist = cos_sim(self.sent2emb[sent], topic_emb)
                dists.append(dist)

        return min(dists)
