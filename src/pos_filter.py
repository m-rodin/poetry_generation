import nltk
from corpy.udpipe import Model

# +
class NLTKTagger:
    def get_pos_tag(self, word):
            token = nltk.word_tokenize(word)
            tag = nltk.pos_tag(token)
            return tag[0][1]
        
class UdpipeTagger:
    def __init__(self, model_path = None):
        if model_path:
            self.model = Model(model_path)
        else:
            raise Exception("You should pass the model")
        
    def get_pos_tag(self, word):
        sent = list(self.model.process(word))[0]
        
        if len(sent.words) != 2:
            print(word, sent.words)
        
        return sent.words[1].xpostag


# -

class POSFiter:
    def __init__(self, words, tagger):
        self.word2pos = self.get_word2pos(words, tagger)
        self.possible_pos_pairs = self.get_possible_pos_pairs()

    def can_follow(self, word1, word2):
        #THESE ARE THE ADVERBS
        #okay_tags = set(["RB","RBR","RBS"])
        #if(tag1==tag2 and tag1 not in okay_tags):
        #    return True
        
        tag1 = self.word2pos[word1]
        tag2 = self.word2pos[word2]

        return tag1 in self.possible_pos_pairs[tag2]
        
    def get_word2pos(self, words, tagger):
        return {word: tagger.get_pos_tag(word) for word in words}
    
    def get_possible_pos_pairs(self):
        #DICTIONARY IS OF FORM (KEY: post_word_pos);(VALUE: pre_word_pos)
        #SO dict["verb"] == set(adverb, noun, ...) BUT NOT set(adjective, determiner, etc)
        pos_list = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS", "LS","MD","NN","NNS","NNP","NNPS", \
                    "PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","TO","UH","VB","VBD","VBG","VBN","VBP", \
                    "VBZ","WDT","WP","WP$","WRB"]
        dictTags = {}
        for tag in pos_list:
            s = set([])
            if("VB" in tag):
                s = set(["CC","RB","RBR","RBS","NN","NN","NNS","NNP","NNPS","MD","PRP"])
                sing_nouns = set(["NN","NNP"])
                plur_nouns = set(["NNS","NNPS"])
                if(tag in set(["VB","VBG","VBP","VBN"])):
                    s.difference(sing_nouns)
                if(tag in set(["VBG","VBZ","VBN"])):
                    s.difference(plur_nouns)
                if(tag in set(["VBG","VBN"])):
                    s.union(set(["VB","VBD","VBP","VBZ"]))
            else:
                s=set(pos_list)
                if("IN"==tag):
                    t = set(["IN","DT","CC"]) #maybe not CC
                    s.difference(t)
                if("JJ" in tag):
                    t = set(["NN","NNS","NNP","NNPS"])
                    s.difference(t)
                if("TO"==tag):
                    t = set(["DT","CC","IN"])
                    s.difference(t)
                if("CC"==tag):
                    t = set(["DT","JJ","JJR","JJS"])
                    s.difference(t)
                if("NN" in tag):
                    t = set(["NN","NNS","NNP","NNPS","PRP","CC"]) #maybe not CC
                    s.difference(t)
                if("MD"==tag):
                    t = set(["DT","VB","VBD","VBG","VBN","VBP","VBZ"])
                    s.difference(t)
                if("PRP"==tag):
                    t = set(["CC","JJ","JJR","JJS","NN","NNS","NNP","NNPS","DT"])
                    s.difference(t)
                if("PRP$"==tag):
                    t = set(["CC","DT","VB","VBD","VBG","VBN","VBP","VBZ","PRP"])
                    s.difference(t)
                adv = set(["RB","RBR","RBS"])
                if(tag not in adv):
                    s.remove(tag)
            dictTags[tag] = s
        return dictTags
