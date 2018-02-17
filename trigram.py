from languageModel import LanguageModel
from collections import defaultdict
import random
import bisect

class TrigramAddOneSmooth(LanguageModel):
    def __init__(self):
        self.trigram_count = defaultdict(int)
        self.word_count = defaultdict(int)
        self.bigram_count = defaultdict(int)
        self.total_trigrams = 0

    def train(self, trainingSentences):
        trigrams = self._get_trigrams(trainingSentences)
        bigrams = self._get_bigrams(trainingSentences)
        for trigram in trigrams:
            self.trigram_count[trigram] += 1
            self.total_trigrams += 1
        for bigram in bigrams:
            self.bigram_count[bigram] += 1

        # Our cases are (UNK, W, W), (W, UNK, W), (W, W, UNK), (UNK, UNK, W)
        # (UNK, W, UNK), (W, UNK, UNK), (UNK, UNK, UNK)
        self.trigram_count[LanguageModel.UNK] += 7

    def _get_trigrams(self, trainingSentences):
        #todo: figure out if I care about the total number of words
        self.total_words = 0
        trigrams = []
        for sent in trainingSentences:
            self.total_words += len(sent)
            for i in range(len(sent)):
                self.word_count[sent[i]] += 1
                if len(sent) < 2:
                    # If it's good enough for the nltk, it's good enough for me
                    continue
                if i == 0:
                    trigrams.append((LanguageModel.START, sent[i], sent[i+1]))
                    self.word_count[LanguageModel.START] += 1
                elif i == len(sent) - 1:
                    trigrams.append((sent[i-1], sent[i], LanguageModel.STOP))
                    self.word_count[LanguageModel.STOP] += 1
                else:
                    trigrams.append(tuple(sent[i:i+3]))
        self.word_count[LanguageModel.UNK] = 7
        return trigrams

    def _get_bigrams(self, trainingSentences):
        bigrams = []
        for sent in trainingSentences:
            for i in range(len(sent)):
                if i == 0:
                    bigrams.append((LanguageModel.START, sent[i]))
                if i <= len(sent) - 2:
                    bigrams.append(tuple(sent[i:i + 2]))
                if i == len(sent) - 1:
                    bigrams.append((sent[i], LanguageModel.STOP))
        return bigrams

    def getWordProbability(self, sentence, index):
        trigram = self._get_trigram(sentence, index)
        bigram = (trigram[0], trigram[1])
        if trigram in self.trigram_count:
            trigram_count = self.trigram_count[trigram]
        else:
            trigram_count = 0
        if bigram in self.bigram_count:
            bigram_count = self.bigram_count[bigram]
        else:
            bigram_count = 0
        return (trigram_count + 1)/(bigram_count + len(self.word_count))

    def _get_trigram(self, sentence, index):
        #In a trigram model, we want the p(word | context), where the context is the
        #previous two words. So if our index is 0, the best we could do is (<S>, word)
        #but that's not enough context for a trigram model
        if index == 0:
            raise ValueError('Index must be >= 1')
        elif index == 1:
            prev_prev_word = LanguageModel.START
            prev_word = sentence[index-1]
            next_word = sentence[index]
        elif index == len(sentence):
            prev_prev_word = sentence[index-2]
            prev_word = sentence[index-1]
            next_word = LanguageModel.STOP
        else:
            prev_prev_word = sentence[index-2]
            prev_word = sentence[index-1]
            next_word = sentence[index]
        return (prev_prev_word, prev_word, next_word)

    def getVocabulary(self, context):
        return self.word_count.keys()

    def generateSentence(self):
        sentence = []
        prev_previous = LanguageModel.START
        #Maybe not the best choice
        previous = random.choice(list(self.word_count.keys()))
        for i in range(20):
            word = self.generateWord(prev_previous, previous)
            if word == LanguageModel.STOP:
                break
            prev_previous = previous
            previous = word
            sentence.append(word)
        return sentence

    def generateWord(self, prev_previous, previous):
        accumulator, num_with_context = self._create_accumulator(prev_previous, previous)
        index = bisect.bisect_left(accumulator, random.randint(1, num_with_context))
        return list(self.word_count.keys())[index]

    def _create_accumulator(self, prev_previous, previous):
        buckets = []
        current_bucket = -1
        occurrences = 0
        for word in self.word_count.keys():
            if self.trigram_count[(prev_previous, previous, word)] == 0:
                current_bucket += 1
            else:
                current_bucket += self.trigram_count[(prev_previous, previous, word)] + 1
                occurrences += self.trigram_count[(prev_previous, previous, word)]
            buckets.append(current_bucket)
        return buckets, (occurrences + len(self.word_count.keys()) - 1)

# if __name__ == '__main__':
#     t = TrigramAddOneSmooth()
#     trainfile = 'data/train-data.txt'
#     with open(trainfile, 'r') as f:
#         trainSentences = [line.split() for line in f.readlines()]
#     t.train(trainSentences)
#     words = defaultdict(int)
#     for i in range(500):
#         word = t.generateWord(LanguageModel.START, "I")
#         words[word] += 1
#     print(sorted(words.items(), key=lambda x : x[1], reverse=True))
#     # print(t.generateSentence())
