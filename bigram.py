from languageModel import LanguageModel
from collections import defaultdict
import bisect
import random

class BigramAddOneSmooth(LanguageModel):
    def __init__(self):
        self.probCounter = defaultdict(float)
        self.bigram_count = defaultdict(int)
        self.word_count = defaultdict(int)  # for use in MLE
        self.total_bigrams = 0

    def train(self, trainingSentences):
        bigrams = self._get_bigrams(trainingSentences)
        for bigram in bigrams:
            self.probCounter[bigram] += 1
            self.total_bigrams += 1
            self.bigram_count[bigram] += 1

        # 3 cases: (word, UNK), (UNK, word), (UNK, UNK)
        self.probCounter[LanguageModel.UNK] += 1
        self.bigram_count[LanguageModel.UNK] += 1

    #So it turns out this method is implementing the add-1 smoothing already
    #But I may still want to modify getWordProbabilityBigram
    def _create_accumulator(self, previous):
        current_bucket = -1
        bucket_list = []
        num_with_previous = 0
        for word2 in self.word_count.keys():
            if self.bigram_count[(previous, word2)] == 0:
                current_bucket += 1
            else:
                current_bucket += self.bigram_count[(previous, word2)] + 1
                num_with_previous += self.bigram_count[(previous, word2)]
            bucket_list.append(current_bucket)
        return bucket_list, (num_with_previous + len(self.word_count.keys()) - 1)

    def _get_bigrams(self, trainingSentences):
        self.total_words = 0
        bigrams = []
        for sent in trainingSentences:
            self.total_words += len(sent)
            for i in range(len(sent)):
                self.word_count[sent[i]] += 1
                if i == 0:
                    bigrams.append((LanguageModel.START, sent[i]))
                    self.word_count[LanguageModel.START] += 1
                if i <= len(sent) - 2:
                    bigrams.append(tuple(sent[i:i + 2]))
                if i == len(sent) - 1:
                    bigrams.append((sent[i], LanguageModel.STOP))
                    self.word_count[LanguageModel.STOP] += 1
        self.word_count[LanguageModel.UNK] = 3  # same three cases as in self.train
        return bigrams

    def getVocabulary(self, context):
        return self.word_count.keys()

    # Maximum likelihood estimate-
    # P(word|previous) = count(previous, word)/count(previous)
    def getWordProbability(self, sentence, index):
        bigram = self._get_bigram(sentence, index)
        if bigram in self.bigram_count:
            bigram_count = self.bigram_count[bigram]
        else:
            bigram_count = 0
        if bigram[0] in self.word_count:
            count_previous = self.word_count[bigram[0]]
        else:
            count_previous = 0
        prob = (bigram_count + 1)/(count_previous + len(self.word_count))
        return prob

    def _get_bigram(self, sentence, index):
        if index == 0:
            previous_word = LanguageModel.START
            next_word = sentence[index]
        elif index == len(sentence):
            previous_word = sentence[index - 1]
            next_word = LanguageModel.STOP
        else:
            previous_word = sentence[index - 1]
            next_word = sentence[index]
        return (previous_word, next_word)

    def generateSentence(self):
        sentence = []
        previous = LanguageModel.START
        for i in range(20): #this is small so as not to throw a memory error
            word = self.generate_word(previous)
            sentence.append(word)
            previous = word
            if word == LanguageModel.STOP:
                break
        return sentence

    #Apparently this method is working as expected, it's just that with so much
    #data and a relatively simple model, you get poor results
    def generate_word(self, previous):
        accumulator, num_with_previous = self._create_accumulator(previous)
        index = bisect.bisect_left(accumulator, random.randint(1, num_with_previous))
        return list(self.word_count.keys())[index]

# if __name__ == '__main__':
#     trainfile = 'data/train-data.txt'
#     with open(trainfile, 'r') as f:
#         trainSentences = [line.split() for line in f.readlines()]
#     b = BigramAddOneSmooth()
#     b.train(trainSentences)

    # print(b.generateSentence())

    # words = defaultdict(int)
    # for i in range(500):
    #     word = b.generate_word(LanguageModel.START)
    #     words[word] += 1
    # print(sorted(words.items(), key=lambda x : x[1], reverse=True))
    # print(b.getWordProbability(["I"], 0))
    # print(b.getWordProbabilityBigram("The other".split(), 1))
    # print(b.generateSentence())