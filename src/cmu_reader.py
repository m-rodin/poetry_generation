# -*- coding: utf-8 -*-
class VoxforgeRuReader:
    def __init__(self, file, **kwargs):
        self.file = file
        
    def get_dicts(self):
        # vowels - гласные звуки
        vowels = ['U', 'O', 'A', 'E', 'Y', 'I']
        # stressed vowel - ударный гласный
        stressed_v = [v + "0" for v in vowels]

        # meters - размер ритм
        word2meters= {}

        # rhyme - рифма
        word2rhyme = {}

        with open(self.file) as f:
            lines = [line.rstrip("\n").split() for line in f]

        for i, line in enumerate(lines):
            # line[0] - слово
            # line[1:] - список слогов

            word = line[0].lower()
            syllables = line[1:]

            # ищем самое правое вхождение в конце ударной гласной (получится что самое длинное слово в приоритете, но так исключают -1)
            r_index = max([''.join(syllables).rfind(v) for v in stressed_v])
            # после этого считаем концовку слова после ударной гласной
            word2rhyme[word] = ''.join(i for i in ''.join(syllables)[r_index:])

            #THIS IF STATEMENT ALLOWS FOR MULTIPLE PRONUNCIATIONS OF A WORD
            if word not in word2meters:
                word2meters[word] = []

            meter = ""
            for syllable in syllables:
                if syllable in vowels:
                    meter += "0"
                if syllable in stressed_v:
                    meter += "1"

            if meter not in word2meters[word]:
                word2meters[word].append(meter)

        return word2meters, word2rhyme


class CMUEngReader:
    def __init__(self, file, **kwargs):
        self.file = file
        
    def get_dicts(self):
        # vowels - гласные звуки
        vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
        # stressed vowel - ударный гласный
        stressed_v = [v + "1" for v in vowels] + [v + "2" for v in vowels]

        # meters - размер ритм
        word2meters= {}

        # rhyme - рифма
        word2rhyme = {}

        with open(self.file, encoding='windows-1252') as f:
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

        return word2meters, word2rhyme
