# -*- coding: utf-8 -*-
##
## Проверяем что word_meter если его вставить на место start_index будет соответствовать паттерну строки
##
def isFitPattern(word_meter, start_index, pattern = '0101010101'):
    if start_index + len(word_meter) > len(pattern):
        return False
    
    for i in range(len(word_meter)):
        if word_meter[i] != pattern[start_index + i]:
            return False

    return True


##
## возвращает словарь в котором позиции в строке соответствуют возможные слова
##
def get_place2words(meter2words, start_index = 0, place2words = {}, pattern = '0101010101'):
    suitable_words_count = 0
    
    # перебираем все доступные слова
    for meter, words in meter2words.items():
        # если meter слова не подходит для места stress, то пропускаем
        if not isFitPattern(meter, start_index, pattern):
            continue
            
        # если подходит, но итоговая длина больше pattern, то пропускаем (на самом деле уже проверяется в isFitPattern)
        end_index = start_index + len(meter)
        if end_index > len(pattern):
            continue
            
        # строка закончена, закругляемся
        if end_index == len(pattern):
            place2words[(start_index, end_index)] = words
            suitable_words_count += 1
            continue            
            
        # слово подошло, но строка еще не закончена, ищем следующее
        place2words, status = get_place2words(meter2words, end_index, place2words, pattern)
        
        # если дальше не нашли продолжения до len(pattern), то пропускаем
        if status == "no_children":
            continue
        
        # рекурсия завершилась, прошли законченные строки с этим словом, сохраняем
        place2words[(start_index, end_index)] = words
        suitable_words_count += 1

    if suitable_words_count == 0:
        return place2words, "no_children"
    
    return place2words, "ok"


class WordSelector:
    def __init__(self, meter2words, pattern):
        self.place2words, _ = get_place2words(meter2words, pattern=pattern)
        
    def get_suitable_words(self, end_index):
        suitable_intervals = []
        for interval in self.place2words.keys():
            if interval[1] == end_index:
                suitable_intervals.append(interval)

        words = []
        for interval in suitable_intervals:
            for word in self.place2words[interval]:
                yield word, interval[0]
