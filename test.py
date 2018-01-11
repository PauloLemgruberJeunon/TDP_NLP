import nltk
from nltk.tag.stanford import CoreNLPPOSTagger

#*******************************************************************************************

# tagged_text = nltk.corpus.treebank.tagged_sents()
# tagged_text = tagged_text[0:150]

raw = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
raw = raw[30:40000]

# tokens = []
# tok_sent = []
# for sents in tagged_text:
#     for (word,tag) in sents:
#         tok_sent.append(word)
#     tokens.append(tok_sent.copy())
#     tok_sent.clear()

tagger = CoreNLPPOSTagger(url='http://localhost:9000')

# tagged_words = []
# tagged_sent = []
# print('To tag sents = ' + str(len(tokens)))
# for i in range(0,len(tokens)):
#     tagged_sent = tagger.tag(tokens[i])
#     tagged_words.append(tagged_sent)
#     print('[Tagging] Current stage = ' + str(i))


# print('Accuracy: ' + str(measurePOSTagAccuracy(tagged_text, tagged_words)))

# quit()



#verb_fd = nltk.FreqDist(word for (word, tag) in tagged_words if tag[0] ==