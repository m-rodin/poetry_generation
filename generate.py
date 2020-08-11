# -*- coding: utf-8 -*-
# %load_ext autoreload
# %autoreload 2

# +
import sys
import yaml
import numpy as np
import itertools
import pickle
import queue as Q
import time
import torch

import argparse

from src.source_prepocess import EngSource, PushkinSource
from src.cmu_reader import CMUEngReader, VoxforgeRuReader
from src.pos_filter import EnPOSFiter, DummyPOSFiter, UdpipeTagger
from src.word_selector import WordSelector
from src.word_embedder import *
from src.sonet import Sonet

from model import LSTMModel


# -

def intersect_words(source_words, embedder, cmu_word2meters, cmu_word2rhyme):
    words = []

    for word, meters in cmu_word2meters.items():
        # оставим только те слова которые есть у поэта
        if word not in source_words:
            continue

        # оставим только те слова у которых есть векторы
        if not embedder.exist_emb(word):
            continue

        words += [word]
    
    words = set(words)
    
    word2meters = dict((word, cmu_word2meters[word]) for word in words)
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
        
    return list(words), word2meters, meter2words, word2rhyme, rhyme2words


def get_meter2words(word2meters):
    meter2words = {}
    for word, meters in word2meters.items():
        for meter in meters:
            if meter not in meter2words:
                meter2words[meter] = set([])
            meter2words[meter].add(word)
    return meter2words


# +
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def sample_topic_rhyme_pair(topic, suitable_words, embedder, rhyme2words, rhyme_used):

    candidate_pairs = []
    for rhyme, words in rhyme2words.items():
        # check rhyme not used already
        if rhyme in rhyme_used:
            continue

        candidate_words = intersection(suitable_words, words)
        candidate_pairs += list(itertools.combinations(candidate_words, 2))

    dist_word2topic = dict(zip(
        suitable_words, [embedder.get_dist(topic, word) for word in suitable_words]
    ))

    # через дистанции считаем странные вероятности по которым потом сэмплируем
    probs = []
    for pair in candidate_pairs:
        if all(dist_word2topic[word] > 0 for word in pair):
            p = max([dist_word2topic[word] for word in pair])**7
        else:
            p = 0
        probs.append(p)

    # norm probs
    probs_sum = np.sum(probs)
    probs = [prob / probs_sum for prob in probs]

    # sample one using probs
    pair_index = np.random.choice(range(len(candidate_pairs)), 1, p=probs)[0]

    return candidate_pairs[pair_index]


# -

def sample_rhyme_pairs(
        topic,
        sonet,
        word_selectors,
        word2rhyme,
        rhyme2words,
        embedder
    ):
    
    rhyme_used = set()
    pairs_picked = []
    
    for pattern in sonet.rhymes:
        gen = word_selectors[pattern].get_suitable_words(len(pattern))
        suitable_words = [w for w, ind in gen]
        
        pair = sample_topic_rhyme_pair(topic, suitable_words, embedder, rhyme2words, rhyme_used)
        
        pairs_picked.append(pair)
        rhyme_used.add(word2rhyme[pair[0]]) 
    
    return pairs_picked


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
    init_prob = 0.
    init_state = model.zero_state(1)
    
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
    
    dist, state = model.next_word(message.sequence[0], message.state, message.prob)
    
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
        #score_adjust = decayRepeat(word, message.sequence, 100 * scale)
        score_adjust = 0
        #length word
        #score_adjust += len(word) * scale / 50
        
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
    state = model.zero_state(1)
    
    for i in range(n_spots):
            #so we're iterating from the rhyme word, but need to feed in forward
            seq = [y for y in reversed(rev_flat_poem[:i+1])]

            p, state = model.next_word(seq[0], state, 0.)
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


# +
def get_inited_class(setting):
    source_class = globals()[setting['class']]
    return source_class(**setting)

def get_instances(settings):
    source = get_inited_class(settings['source'])
    embedder = get_inited_class(settings['embedder'])
    cmu_reader = get_inited_class(settings['cmu'])
    
    if 'tagger' in settings['pos_filter']:
        tagger = get_inited_class(settings['pos_filter']['tagger'])
    else:
        tagger = None
        
    pos_filter = get_inited_class({**settings['pos_filter'], 'tagger': tagger})

    return source, embedder, cmu_reader, pos_filter


# -

def generate(args):
    # system arguments
    topic = args.topic
    
    # seed for reproducibility
    np.random.seed(args.seed)
    
    with open(args.settings) as f:
        settings = yaml.load(f, Loader=yaml.BaseLoader)

    source, embedder, cmu_reader, pos_filter = get_instances(settings)

    source_words = source.get_words()
    source_texts = source.get_texts()
    
    cmu_word2meters, cmu_word2rhyme = cmu_reader.get_dicts()

    words, word2meters, meter2words, word2rhyme, rhyme2words = intersect_words(
        source_words, embedder, cmu_word2meters, cmu_word2rhyme
    )
    
    pos_filter.init_dicts(words)
        
    #cmu_meter2words = get_meter2words(cmu_word2meters)
    #cmu_words = list(cmu_word2meters.keys())
    
    sonet = Sonet(**settings['sonet'])
    
    word_selectors = {}
    for line_pattern in set(sonet.line_patterns):
        # здесь скорее всего надо cmu_meter2words
        word_selectors[line_pattern] = WordSelector(meter2words, line_pattern)
    
    width = 20
    
    rhyme_pairs = sample_rhyme_pairs(
        topic,
        sonet,
        word_selectors,
        word2rhyme,
        rhyme2words,
        embedder
    )
    last_words = sonet.make_author_order(rhyme_pairs)
    
    model_dir = settings['params']['model_dir']
    
    # загрузка модели
    with open(model_dir + '/vocab.pkl', 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
    
    model = LSTMModel(len(vocab), 300, 1000, 3)
    model.load_state_dict(torch.load(model_dir + '/model-last.pth'))
    model.set_vocab(vocab)
    model.eval()
    
    poem = []
    
    # для каждого слова ищем строчки
    for line_i, last_word_in_line in enumerate(last_words):
        line_pattern = sonet.line_patterns[line_i]
        word_selector = word_selectors[line_pattern]
        
        meter_len = len(word2meters[last_word_in_line][0])

        end_index = len(line_pattern) - meter_len

        lines = search_line_variants(
                model,
                [last_word_in_line],
                end_index,
                word_selector,
                pos_filter,
                width,
            )
        
        line = sampleLine(lines)

        poem.append(line)
        
    poem_p = postProcess(poem, model)

    with open("./output_poems/%s.txt" % topic, "w") as text_file:
        print("(saved in output_poems)")
        print("\n topic:" + topic, "\n\n" + poem_p)
        text_file.write(topic + "\n\n" + poem_p)


if(__name__ == "__main__"):

    parser = argparse.ArgumentParser()
    
    parser.add_argument('topic', metavar='topic', type=str, help='sonet topic')
    parser.add_argument('--settings', type=str, default="settings/pushkin.yaml", help='file containing settings')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    args = parser.parse_args()
    generate(args)
