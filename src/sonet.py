# +
import re

class Sonet:
    def __init__(self, rhyme_pattern = "AbAb CCdd EffE gg", stress_pattern = '01', feet_number = 4, topic_pairs = 5):
        # only letters
        rhyme_pattern = re.sub('[^a-zA-Z]+', '', rhyme_pattern)
        feet_number = int(feet_number)
        
        self.line_patterns, self.rhymes = self.get_line_patterns(rhyme_pattern, stress_pattern, feet_number)
        
        self.rhyme_map = self.get_rhyme_map(rhyme_pattern)        
        
    def get_line_patterns(self, rhyme_pattern, stress_pattern, feet_number):
        patterns = []
        rhymes = []
        
        used = []
        for ch in rhyme_pattern:
            if ch.isupper():
                line_pattern = stress_pattern * feet_number
            else:
                line_pattern = stress_pattern * feet_number + '0'
                
            patterns.append(line_pattern)
            
            if ch not in used:
                used.append(ch)
                rhymes.append(line_pattern)
                
        return patterns, rhymes
    
    def _find(self, s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def get_rhyme_map(self, rhyme_pattern):
        rhyme_map = [0] * len(rhyme_pattern)
        
        rhymes_chars = set(rhyme_pattern.lower())
        rhymes_chars = sorted(list(rhymes_chars))
        
        for pair_i, ch in enumerate(rhymes_chars):
            inds = self._find(rhyme_pattern.lower(), ch)

            for j, ind in enumerate(inds):
                rhyme_map[ind] = pair_i * 2 + j % 2

        return {i: rhyme_map[i] for i in range(len(rhyme_pattern))}
    
    def make_author_order(self, rhyme_pairs):
        flat_list = [word for pair in rhyme_pairs for word in pair]
        
        return [flat_list[self.rhyme_map[i]] for i in range(len(flat_list))]
