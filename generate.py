# -*- coding: utf-8 -*-
# %load_ext autoreload
# %autoreload 2

# +
import sys
import numpy as np
import itertools
import pickle
import queue as Q
import time

from text_prepocessing import simple_clean
from pos_filter import POSFiter
from word_selector import WordSelector

from gensim.models.keyedvectors import KeyedVectors


# -

def get_cmu_dicts(cmu_path = "./cmudict-0.7b.txt"):
    # vowels - гласные звуки
    vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
    # stressed vowel - ударный гласный
    stressed_v = [v + "1" for v in vowels] + [v + "2" for v in vowels]

    # meters - размер ритм
    word2meters= {}

    # syllables - слоги
    word2syllables = {}

    # rhyme - рифма
    word2rhyme = {}

    with open(cmu_path, encoding='windows-1252') as f:
        lines = [line.rstrip("\n").split() for line in f if (";;;" not in line)]
    
    word2meters["i"] = ["0"]
    word2meters["the"] = ["0"]
    
    for i, line in enumerate(lines):
        # line[0] - слово
        # line[1:] - список слогов

        word = line[0].lower()
        syllables = line[1:]
                
        # если есть () одновременно, то скорее всего в конце (1), удалим 
        if "(" in word and ")" in word:
            word = word[:-3]

        # syllables, оставляем чисты слоги, фильтруем от ударений
        word2syllables[word] = [''.join(i for i in syllable if not i.isdigit()) for syllable in syllables]

        # ищем самое правое вхождение в конце ударной гласной (получится что самое длинное слово в приоритете, но так исключают -1)
        r_index = max([''.join(syllables).rfind(v) for v in stressed_v])
        # после этого считаем концовку слова после ударной гласной
        word2rhyme[word] = ''.join(i for i in ''.join(syllables)[r_index:] if not i.isdigit())

        #THIS IF STATEMENT ALLOWS FOR MULTIPLE PRONUNCIATIONS OF A WORD
        if word not in word2meters:
            word2meters[word] = []

        meter = ""
        for syllable in syllables:
            for char in syllable:
                if char in "012":
                    if char == "2":
                        meter += "1"
                    else:
                        meter += char
        
        if meter not in word2meters[word]:
            word2meters[word].append(meter)
            
    return word2meters, word2syllables, word2rhyme


def get_source_texts(texts_path):
    texts = open(texts_path)
    texts = texts.read()
    
    words = set([simple_clean(word) for word in texts.split()])
    
    return words, texts


def intersect_words(source_words, glove_words, cmu_word2meters, cmu_word2syllables, cmu_word2rhyme):
    words = []

    for word, meters in cmu_word2meters.items():
        # оставим только те слова которые есть у поэта
        if word not in source_words:
            continue

        # оставим только те слова у которых есть glove векторы
        if word not in glove_words:
            continue

        # оставляем только слова с правильным ритмом
        for meter in meters:
            if meter == '1' * (len(meter) % 2) + '01' * (len(meter) // 2):
                words += [word]
    
    words = set(words)
    
    word2meters = dict((word, cmu_word2meters[word]) for word in words)
    word2syllable = dict((word, cmu_word2syllables[word]) for word in words)
    word2rhyme = dict((word, cmu_word2rhyme[word]) for word in words)

    rhyme2words = {}    
    for word, rhyme in word2rhyme.items():
        if rhyme not in rhyme2words:
            rhyme2words[rhyme] = []
        rhyme2words[rhyme].append(word)
        
    meter2words = {}
    for word, meters in word2meters.items():
        for meter in meters:
            if meter not in meter2words:
                meter2words[meter] = set([])
            meter2words[meter].add(word)
        
    return list(words), word2meters, meter2words, word2syllable, word2rhyme, rhyme2words


def get_meter2words(word2meters):
    meter2words = {}
    for word, meters in word2meters.items():
        for meter in meters:
            if meter not in meter2words:
                meter2words[meter] = set([])
            meter2words[meter].add(word)
    return meter2words


# +
def cos_sim(w1, w2):
    return np.dot(w1, w2) / np.linalg.norm(w1) / np.linalg.norm(w2)

def calculate_mean_glove(words):
    return np.mean([glove_model[word] for word in words], axis=0)


# -

def sample_rhyme_pairs(topic, words, word2rhyme, rhyme2words, model_vocab, glove_model, topic_pairs = 5, common_pairs = 2):
    
    topic_glove = calculate_mean_glove(topic.split())
    
    # дистанция для каждого слова из текстов до prompt
    dist_word2topic = dict(zip(
        words, [cos_sim(topic_glove, glove_model[word]) for word in words]
    ))
    
    rhyme_used = set()
    
    pairs_picked = []
    
    for i in range(topic_pairs):
        # candidate pairs for this round
        # все пары что есть в rhyme2words засунем в pairs (сэмплируем комбинацию длиной 2)
        pairs = []
        for rhyme, words in rhyme2words.items():
            # check rhyme not used already
            if rhyme not in rhyme_used:
                pairs += list(itertools.combinations(words, 2))
                
        # через дистанции считаем странные вероятности по которым потом сэмплируем
        probs = []
        for pair in pairs:
            if all(dist_word2topic[x] > 0 for x in pair):
                p = max([dist_word2topic[x] for x in pair])**7
            else:
                p = 0
            probs.append(p)
        
        # norm probs
        probs_sum = np.sum(probs)
        probs = [prob / probs_sum for prob in probs]
        
        # sample one using probs
        pair_index = np.random.choice(range(len(pairs)), 1, p=probs)[0]
        
        pairs_picked.append(pairs[pair_index])
        first_word = pairs[pair_index][0]
        rhyme = word2rhyme[first_word]
        rhyme_used.add(rhyme)

    # common rhyme pairs
    alphabet = set("abcdefghijklmnopqrstuvwxyz")
    common = pickle.load(open("./CommonRhymes.pkl", "rb"))
    
    # дернули топ 50 рифм (тут какой-то тупой фильтр на алфавит, выглядит бесполезным)
    pairs = [x[0] for x in common.most_common(50) if all(word not in alphabet for word in x[0])]

    good_pairs = []
    for pair in pairs:
        if pair[0] not in word2rhyme:
            continue
        if word2rhyme[pair[0]] in rhyme_used:
            continue
#        if pair[0] not in model_vocab:
#            continue
#        if pair[1] not in model_vocab:
#            continue

        good_pairs.append(pair)
    
    common_indexes = np.random.choice(len(good_pairs), common_pairs)
    pairs_picked += [pairs[index] for index in common_indexes]

    return pairs_picked


# пересортировывает рифмованные пары в соответствии с тем как надо в результате
def make_author_order(rhymes, order = ''):
    rhymes = np.array(rhymes).reshape(-1)

    index = np.array([
            [0,2],
            [1,3],
            [4,6],
            [5,7],
            [8,10],
            [9,11],
            [12,13]
        ]).reshape(-1)

    return rhymes[index]


class Message:
    def __init__(self, prob, state, sequence, end_index):
        self.prob = prob
        self.state = state
        self.sequence = sequence
        self.end_index = end_index
        
    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return False


def search_line_variants(
    model,
    sequence,
    end_index, # [last_word_in_line], end_index
    word_selector,
    pos_filer,
    beam_width, # width
):
    masterPQ = Q.PriorityQueue()
    
    #initial case
    init_prob, init_state = model.init_values()
    
    body = Message(init_prob, init_state, sequence, end_index)

    masterPQ.put((init_prob, body))
    
    completed_lines = []
    explored = set([])
    
    while not masterPQ.empty():
        depthPQ = Q.PriorityQueue()
        
        while not masterPQ.empty():
            priopity, message = masterPQ.get()

            deeper_messages = beamSearchOneLevel(model, message, pos_filer, word_selector)
                
            for priopity, deeper_message in deeper_messages:
                # need to make sure each phrase is being checked uniquely
                # (want it to be checked once in possible branches then never again)
                sequence_tup = tuple(deeper_message.sequence)
                
                if sequence_tup in explored:
                    continue
                    
                if deeper_message.end_index == 0:
                    completed_lines += [deeper_message]
                    continue
                
                explored.add(sequence_tup)
                depthPQ.put((priopity, deeper_message))

                # оставляем только beam_width вариантов (т.е. откидываем с конца что уже не попало)
                if depthPQ.qsize() > beam_width:
                    depthPQ.get()
        masterPQ = depthPQ

    return completed_lines


def decayRepeat(word, sequence, scale):
    score_adjust = 0
    decr = -scale
    for w in range(len(sequence)):
        if word == sequence[w]:
            score_adjust += decr
            
        #decreases penalty as the words keep getting further from the new word
        decr += scale / 10
    return score_adjust


##
## перебирает возможные слова из dictWordTransitions. возвращает подходящие с вероятностями
## сетка предсказывает предыдущее слово
##
# scale is the significant magnitude required to affect the score of bad/good things
#
def beamSearchOneLevel(model, message, pos_filer, word_selector, scale = .02):
    result = []
    
    dist, state = model.predict(message)
    
    for word, start_index in word_selector.get_suitable_words(message.end_index):
        
        #PREVENTS REPEAT ADJACENT WORDS OR PROBLEM-TAGGED WORDS
        # 0 - это предыдущее слово
        if word == message.sequence[0]:
            continue
            
        # ограничение на последовательность POS, правила
        if not pos_filer.can_follow(word, message.sequence[0]):
            continue
            
        #CALCULATES ACTUAL SCORE
        key = model.get_token(word)
        new_prob = dist[key]
        
        #FACTORS IN SCORE ADJUSTMENTS
        #repeats
        score_adjust = decayRepeat(word, message.sequence, 100 * scale)
        #length word
        score_adjust += len(word) * scale / 50
        
        m = Message(new_prob + score_adjust, state, [word] + message.sequence, start_index)

        result.append((m.prob, m))

    return result


def sampleLine(lines):
    lines.sort(key=lambda x: x.prob, reverse = True)
    
    top_k = min(10, len(lines))
    
    probs = list(map(lambda m: np.exp(m.prob), lines[:top_k]))    
    probs = np.exp(probs) / sum(np.exp(probs))
    
    index = np.random.choice(range(top_k), 1, p=probs)[0]
    
    return lines[index].sequence


class ModelMock:
    def __init__(self):
        pass
    
    def init_values(self):
        return np.array([[0]]), [1,2,3] # init_prob, init_state
    
    def predict(self, message):
        return [np.random.uniform()], [1,2,3]
    
    def get_token(self, word):
        return 0


def postProcess(poemOrig, model):
    poem = [[e for e in l] for l in poemOrig]
    ret = ""
    
    '''find comma locations here, ith line'''
    flat_poem = []
    for line in poem:
        for word in line:
            flat_poem.append(word)
            
            
    rev_flat_poem = [x for x in reversed(flat_poem)]
    n_spots = len(rev_flat_poem) - 1
    
    comma_probs = np.zeros(n_spots)    
    
    for i in range(n_spots):
            seq = [y for y in reversed(rev_flat_poem[:i+1])] #so we're iterating from the rhyme word, but need to feed in forward
            
            p, state = model.predict(1)
            comma_probs[i] = p[model.get_token(",")]
            
    comma_probs = np.array([y for y in reversed(comma_probs)])
    
    num_commas = int(np.random.normal(9,2))
    num_commas = min(num_commas,14)
    num_commas = max(3,num_commas)
    
    cut_prob = [i for i in reversed(sorted(comma_probs))][num_commas]
    
    spots = np.argwhere(comma_probs > cut_prob)
    comma_counter = 0
    
    '''put some commas in there'''
    for i in range(14):
        for j in range(len(poem[i])):
            if(comma_counter in spots.squeeze()):
            #if( comma_counter in spots.squeeze() and (not i%3 or j=):
                poem[i][j] = poem[i][j] + ","
            comma_counter = comma_counter + 1

    ''' capitalize and print'''    
    for i in range(14):
        line = poem[i]
        for j in range(len(line)):
            if line[j] == "i":
                line[j] = "I"
        line[0] = str.upper(line[0][0]) + line[0][1:]
        if(i == 3 or i == 7 or i == 11 or i == 13):
            if("," in line[-1]):
                line[-1] = line[-1][:-1]
            ret += ' '.join(line) + "." + '\n'
        else:
            ret += ' '.join(line) + '\n'

    return ret

if(__name__ == "__main__"):
    
    start_time = time.time()

    # system arguments
    topic = sys.argv[1]
    
    # seed for reproducibility
    try:
        seed = int(sys.argv[2])
    except:
        seed = 1
    np.random.seed(seed)
    
    glove_model = KeyedVectors.load_word2vec_format('./glove/glove.6B.300d.word2vec.txt', binary=False)
    glove_words = glove_model.vocab.keys()

    source_words, source_texts = get_source_texts('../poetry-generation/data/whitman/input.txt')

    cmu_word2meters, cmu_word2syllables, cmu_word2rhyme = get_cmu_dicts('./cmudict-0.7b.txt')

    words, word2meters, meter2words, word2syllable, word2rhyme, rhyme2words = intersect_words(
        source_words, glove_words, cmu_word2meters, cmu_word2syllables, cmu_word2rhyme
    )
    
    cmu_meter2words = get_meter2words(cmu_word2meters)
    cmu_words = list(cmu_word2meters.keys())
    
    model = ModelMock()
    
    line_pattern = "0101010101"
    sonnet_pattern = "ABAB CDCD EFEF GG"
    
    # здесь скорее всего надо cmu_meter2words
    word_selector = WordSelector(meter2words, line_pattern)
    pos_filer = POSFiter(words)
    
    print("\nDICT PREPARITIONS", time.time() - start_time)
    start_time = time.time()
    
    width = 20
    
    rhyme_pairs = sample_rhyme_pairs(topic, words, word2rhyme, rhyme2words, {}, glove_model)
    last_words = make_author_order(rhyme_pairs, sonnet_pattern)
    
    print("\nRHYMES", time.time() - start_time)
    start_time = time.time()
    
    poem = []
    
    # для каждого слова ищем строчки
    for last_word_in_line in last_words:
        meter_len = len(word2meters[last_word_in_line][0])

        end_index = len(line_pattern) - meter_len

        lines = search_line_variants(
                model,
                [last_word_in_line],
                end_index,
                word_selector,
                pos_filer,
                width,
            )

        line = sampleLine(lines)

        poem.append(line)
        
    print("\nLINES", time.time() - start_time)
        
    poem_p = postProcess(poem, model)

    with open("./output_poems/%s.txt" % topic, "w") as text_file:
        print("(saved in output_poems)")
        print("\n", topic, "\n", poem_p)
        text_file.write(topic + "\n\n" + poem_p)
