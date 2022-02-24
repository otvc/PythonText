import unittest
import collections
import sys


sys.path.append('./TransContainer')
from Tokenizers import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.bpe = BPETokenizer()

    def test_split_sequence_by_subseqs(self):
        sequence = 'Hello, i\'m Oleg Radaev. He-he-he'
        a = ['Hello', ',', 'i', '\'', 'm', 'Oleg', 'Radaev', '.', 'He', '-', 'he', '-', 'he']
        self.assertCountEqual(self.bpe.split_sequence_by_subseqs(sequence), a)

    def test_get_ngrams_from_seq(self):
        sequence = 'Hello, i\'m Oleg Radaev. He-he-he'
        t_ngrams =  [('H', 'e'), ('e', 'l'),  ('l', 'l'), ('l', 'o'), ('o', ','),  (',', ' '),  (' ', 'i'),  ('i', "'"),  ("'", 'm'),
                    ('m', ' '),  (' ', 'O'),  ('O', 'l'),  ('l', 'e'), ('e', 'g'),  ('g', ' '),  (' ', 'R'),  ('R', 'a'),  ('a', 'd'), 
                    ('d', 'a'),  ('a', 'e'),  ('e', 'v'), ('v', '.'), ('.', ' '),  (' ', 'H'),  ('H', 'e'),  ('e', '-'),  ('-', 'h'),
                    ('h', 'e'),  ('e', '-'),  ('-', 'h'), ('h', 'e')]
        self.assertCountEqual(self.bpe.get_ngrams_from_seq(sequence), t_ngrams)

    def test_add_new_ngrams(self):
        
        sequence = 'Hello, i\'m Oleg Radaev. He-he-he'
        ngrams = self.bpe.get_ngrams_from_seq(sequence)
        test_dict = {(' ', 'H'): 1, (' ', 'O'): 1, (' ', 'R'): 1, (' ', 'i'): 1, ("'", 'm'): 1, (',', ' '): 1, ('-', 'h'): 2, ('.', ' '): 1,
                    ('H', 'e'): 2, ('O', 'l'): 1, ('R', 'a'): 1, ('a', 'd'): 1, ('a', 'e'): 1, ('d', 'a'): 1, ('e', '-'): 2, ('e', 'g'): 1,
                    ('e', 'l'): 1, ('e', 'v'): 1, ('g', ' '): 1, ('h', 'e'): 2, ('i', "'"): 1, ('l', 'e'): 1, ('l', 'l'): 1, ('l', 'o'): 1,
                    ('m', ' '): 1, ('o', ','): 1, ('v', '.'): 1}
        inputed_dict = collections.defaultdict(int)
        inputed_dict = self.bpe.add_new_ngrams(inputed_dict, ngrams)
        self.assertCountEqual(inputed_dict, test_dict)

    def test_alphabet(self):
        sequence = 'Hello, He-he-he'.lower()
        test_alphabet = ['h','e', 'o','l','-', ',', ' ']
        alphabet = self.bpe.get_alphabet(sequence)

        self.assertCountEqual(alphabet, test_alphabet)

if __name__ == '__main__':
    unittest.main()