# -*- coding: utf-8 -*-
import re
import nltk


class EngSource:
    def __init__(self, file, **kwargs):
        texts = open(file)
        texts = texts.read()
        self.texts = texts
        
    def get_texts(self):
        return self.texts
    
    def get_words(self):
        return set([simple_en_clean(word) for word in self.texts.split()])


class PushkinSource:
    def __init__(self, file, **kwargs):
        with open(file) as f:
            lines = f.readlines()
            
        texts = []
        text = ''

        el_counter_prev = 0
        for i, line in enumerate(lines):
            if line == '\n':
                el_counter = el_counter_prev + 1
            else:
                el_counter = 0

            if el_counter_prev > 3 and el_counter == 0:
                texts.append(text)
                text = ''

            if line != '\n':
                if len(text):
                    text += " "
                text += line[:-1]

            el_counter_prev = el_counter
        
        self.texts = texts
        
    def get_texts(self):
        return self.texts
    
    def get_words(self):
        texts = [pushkin_clean(text) for text in self.texts]
        
        all_tokens = []
        for text in texts:
            all_tokens += nltk.word_tokenize(text)

        return set(all_tokens)


# +
def simple_en_clean(string):
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"-", "", string)
        string = re.sub(r":", "", string)

        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'d", "ed", string) #for the old style
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"I\'ve", "I have", string)
        string = re.sub(r"\'ll", " will", string)

        string = re.sub(r"[0-9]+", "EOS", string) # EOS tag for numeric titling

        string = re.sub(r";", ",", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\.", " . ", string)

        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

def pushkin_clean(text):
    text = re.sub(r"(\[[0-9]+\])", "", text) # remove notes
    text = re.sub("<\?>", "", text) # remove <?>
    text = re.sub(r"[\[\]\<\>]+", "", text) # remove all brackets
    text = re.sub("…", ".", text)

    return text.strip().lower()
