import math

'''
Tuan Do
'''
class LanguageModel(object) :

    START = "<S>"
    STOP = "</S>"
    UNK = "*UNKNOWN*"

    '''
    Constructs a language model from a collection of sentences.
    -----
    
    trainingSentences = list of lists of strings (words)
    '''
    def train(self, trainingSentences):
        pass

    '''
    Returns the probability, according to the model, of the word specified
      by the argument sentence and index. Index ranges from 0 to len(sentence),
      inclusive. If index==len(sentence), return P(STOP | context).
      
    -----
    sentence: list of strings (words)
    index: index to calculate the probablity
    '''
    def getWordProbability(self, sentence, index):
        pass


    '''
    Returns the set of tokens the model makes predictions for. This
    includes STOP and UNK, but not START (because we do not need to
    compute P(START | context)).

    -----
    Return: list of strings
    '''
    def getVocabulary(self, context):
        pass

    '''
    Returns a random sentence sampled according to the model.
    
    -----
    Return: list of strings
    '''
    def generateSentence(self):
        pass

    #-----------------------------------------------------------------------

    '''
    Returns the probability, according to the model, of the specified
    sentence.  This is the product of the probabilities of each word in
    the sentence (including a final stop token).
    
    -----
    sentence: list of strings
    '''
    def getSentenceLogProbability(self, sentence) :
        logProbability = sum( math.log(self.getWordProbability(sentence, i), 2) for i in range(len(sentence) + 1)) 
        return logProbability
    

    '''
    Given a list of words, sums over the probabilities of every token that
    could follow. If the model implements a valid probability
    distribution, this should always sum to 1.
    '''
    def checkProbability(self, context):
        modelsum = 0.0;
        for token in self.getVocabulary(context):
            context.append(token)
            modelsum += self.getWordProbability(context, len(context) - 1)
            del context[-1]

        return modelsum
