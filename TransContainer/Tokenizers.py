import re
import collections
import copy
import operator


#Function for sorting input
def custom_insort(items, x, key = None, cmp_func = None):
    if not items:
        return [x]
    if cmp_func is None:
      cmp_func = lambda x, y: x >= y
    if key is None:
      key = lambda x: x
    for i in range(len(items)):
        if cmp_func(key(items[i]), key(x)):
            items.insert(i, x)
            return items
    return items + [x]


class BPETokenizer():
  def __init__(self, lower_case = True) -> None:
      self.lower_case = lower_case
      self.reg_for_split = re.compile('((!|"|#|$|%|&|\'|\(|\)|\*|\+|,|-|\.|/|:|;|<|=|>|\?|@|[|\\|]|\^|_|`|{|\||}|~|…|»|«|—)|\\w+)')
      self.dict_with_ngrams = collections.defaultdict(int)
      self.vocab = []
      self.corpus = {}

  '''
  Function for getting actual vocabulary from sequences. Function which finding
  popular n-gram in sequences and append this in list if vocabulary.
  And this vocabulary will use to encoding.
    Params:
      sequences:list - list of text for training
      vocab_max:int - is maximum size of the vocabularity from tokenization process

  Return vocabularity
  '''
  def train_tokenizer(self, sequences:list, vocab_max:int = 30):
      #get all words from sequences
      words = self.split_list_sequences(sequences)
      #get alphabet from sequences
      alphabet = self.get_alphabet(sequences)
      #init dict, which contain words and frequances
      words_freqs = collections.defaultdict(int)
      words_freqs = self.add_new_ngrams(words_freqs, words)
      words_ngrams = self.words_ngrams_dict(words_freqs)
      #get ngram corpus:
      self.corpus = self.get_corpus_pf(words_ngrams, words_freqs)
      temp_vocab, self.corpus = self.__train_loop(alphabet,
                                                  words_freqs,
                                                  words_ngrams,
                                                  self.corpus,
                                                  vocab_max)
      self.vocab.extend(temp_vocab)
      return self.vocab

  '''
  Function for training vocabulary.
    Params:
      vocab: list - is a list of vocabularity, is can be null or contain alphabet and so on
      words_freqs: dict - is a dictionary, where key is specific word and value is frequences specific word in text
      words_ngrams: dict - is a dictionary, where key is a specific word and value is ngrams which make up specific word 
      corpus: dict - is a dictionary, where key is specific ngram and value is frequences of specific ngram in text
      vocab_max: int - is maximum size of the vocabularity from encoding process

  Return vocabulary of tokens, and new corpus
  '''
  def __train_loop(self, vocab, words_freqs, words_ngrams, corpus, vocab_max:int):
    while len(vocab) < vocab_max:
      #Select the most popular par from text corpus
      best_pair = self.max_dict_value(corpus)[0]
      #Merging best pair and change ngrams in all wards, which contain the best pair
      for word in words_ngrams:
        ngrams = words_ngrams[word]
        if len(ngrams) == 1:
            continue
        i = 0
        while i < len(ngrams) - 1:
          if ngrams[i] == best_pair[0] and ngrams[i + 1] == best_pair[1]:
            ngrams = ngrams[:i] + [best_pair[0] + best_pair[1]] + ngrams[i+2:]
          else:
            i+=1
        words_ngrams[word] = ngrams
      #Becouse we change words_ngrams dictionary, which words translate to ngrams, then we should
      #corpus of ngrams, becouse we got new ngrams in words
      corpus = self.get_corpus_pf(words_ngrams, words_freqs)
      #append to vocab
      vocab.append(best_pair[0] + best_pair[1])
    return vocab, corpus
  
  '''
  Function, which split sequences to tokens
    Params:
      sequences: list - list of text for tokenization
  Return - list of tokens-list according sequences
  '''
  def tokenization(self, sequences):
    if len(self.vocab) == 0:
      raise Exception('Before using tokenization you should generate vocabularity with function "train_tokenizer"')
    
    vocab = sorted(self.vocab, key = lambda x: len(x), reverse = True)
    vocab_len = len(vocab)

    #list of tokens-lists for sequences accordingly
    tokens_text = []

    sorted_corpus = self.sort_ngrams_dict(self.corpus)
    
    for i, sequence in enumerate(sequences, start = 0):
      tokens_in_seq = []
      words = self.split_sequence_by_subseqs(sequence)
      #Run for all ngram which contain, up popuar to common
      #Find it in a words and split a word on ngrams
      for word in words:
        
        #list which contain ngram and index ngram in word from  
        ngrams_pos = []
        #index indicating ngram in vocabulary list 
        ngram = 0 
        
        while ngram < vocab_len:
          #while word is not empty
          l = len(word)
          while word != '#' * l:
            idx = word.find(vocab[ngram])
            if idx != -1:
              #last ngram index in word
              end = idx + len(vocab[ngram])
              #set ngram and, index from which it start, for sorting after translation all word
              temp = (vocab[ngram], idx)
              ngrams_pos = custom_insort(ngrams_pos, temp, key = lambda x: x[1])
              #delete ngram from word
              word = word[:idx]+ '#' * len(vocab[ngram]) + word[end:]
            ngram = (ngram + 1) % vocab_len
          break
        tokens_in_seq.append([ngram[0] for ngram in ngrams_pos])
      tokens_text.append(tokens_in_seq)
      
    return tokens_text
    
  '''
  Function which getting generator, which run in sequences end return subseq in sequens
  '''
  def get_gen_for_split(self, sequences):
    def gen_split():
      for sequence in sequences:
        yield self.split_sequence_by_subseqs(sequence)
    return gen_split

  '''
  Function for split sequences to subseq.
    Params:
      sequences: list - list of sequences to split

  Return lists with according subsequences
  '''
  def split_list_sequences(self, sequences: list):
    sub_sequences = []
    for sequence in sequences:
      sub_sequences.extend(self.split_sequence_by_subseqs(sequence))
    return sub_sequences
  '''
  Function for split sequence to subseq.
    Params:
      sequences: list - list of subsequences

  Return lists with according subsequences
  '''
  def split_sequence_by_subseqs(self, sequence: str):
    '''
    In lambda extension into map function return x[0], becouse findall return tuple of (value, '')
    '''
    sequence = list(
      map(lambda x: x[0],
      re.findall(self.reg_for_split, sequence))
    )
    return sequence[:-1]

  '''
  Function which split sequences to bigrams
    Params:
      sequence: str - is sequence to get ngrams in sequence

  Return ngrams from sequence
  '''
  def get_ngrams_from_seq(self, sequence: str, size:int = 1):
    ngrams = []
    for i in range(0, len(sequence) - 1):
      ngrams.append((sequence[i], sequence[i+1]))
    return ngrams

  '''
  Generate dictionary from words to ngram which contains in word
  
  Return dictionary, where key is a specific word and value is ngrams which make up specific word
  
  Example:
    Input:
      bpe = BPETokenizer()
      words = {'data': 3}
      bpe.words_ngram_dict(words)
    Output:
      {
        'data': ['da', 'ta']
      }
  '''
  def words_ngrams_dict(self, word_freqs:dict):
    return {word: [c for c in word] for word in word_freqs.keys()}

  '''
  Generate corpus of ngrams and count in words_ngrams

  Return dictionary, where key is specific word and value is frequences specific word in text
  
  Example:
    Input:
      bpe = BPETokenizer()
      word_ngram = {'data': ['d', 'a', 'ta']}
      word_freq = {'data': 3}
      bpe.get_corpus_pf(word_ngram, word_freq)
    Output:
      {
        'd': 3,
        'a': 3,
        'ta': 3
      }
  '''
  def get_corpus_pf(self, words_ngrams:dict, words_freqs: dict):
    corpus_pf = collections.defaultdict(int)

    for word, freq in words_freqs.items():
        ngrams = words_ngrams[word]
        if len(ngrams) == 1:
            continue
        for i in range(len(ngrams) - 1):
            pair = (ngrams[i], ngrams[i + 1])
            corpus_pf[pair] += freq
    return corpus_pf

  '''
  Function which added new ngram sequence in dict
 
  '''
  def add_new_ngrams(self, dict_seq: dict, ngrams: list):
    for ngram in ngrams:
      dict_seq[ngram] += 1
    return dict_seq

  '''
  Function, returning alphabet from inputed sequences
    params:
      sequences - is list of the string values

  Return all the symbols from sequences
  '''
  def get_alphabet(self, sequences: list):
    alphabet = []
    for sequence in sequences:
      for letter in set(sequence):
        if letter not in alphabet:
          alphabet.append(letter)
    return alphabet

  '''
  Function returning ngrams dict
  '''
  def get_ngrams_dict(self):
    return copy.copy(self.dict_with_ngrams)

  '''
  Function which return sorted dictionary
  '''
  def sort_ngrams_dict(self, dict_with_ngrams):
    return {key: value 
                             for key, value in sorted(dict_with_ngrams.items(), 
                                                      key = lambda item: item[1],
                                                      reverse = True)}
  '''
  Function for recieving maximum value from dict
    Params:
      data: dict - is a dictionary, where finding maximum value
  
  Return item from inputed dict 
  '''
  def max_dict_value(self, data:dict, key = operator.itemgetter(1)):
    return max(data.items(), key=key)