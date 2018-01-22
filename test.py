import nltk
import utils
from nltk.tag.stanford import CoreNLPPOSTagger

#*******************************************************************************************

# tagged_text = nltk.corpus.treebank.tagged_sents()
# tagged_text = tagged_text[0:150]

a = [1,2,3]
a.pop()
# raw = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
# raw = raw[30:40000]
sents = nltk.corpus.treebank.tagged_sents()

i = 0
filtered_sents = []
while i < len(sents):
    j = 0
    while j < len(sents[i]):
        if sents[i][j][1] == '-NONE-':
            # print('NONE')
            del sents[i][j]
            j -= 1
        elif sents[i][j][0].startswith('*'):
            del sents[i][j]
            j -= 1
        elif sents[i][j][0] == '1\/2':
            del sents[i][j]
            j -= 1
        j += 1
    filtered_sents.append(sents[i])
    i += 1

filtered_sents = filtered_sents[0:400]

tokens = []
for sent in filtered_sents:
    for word,tag in sent:
        tokens.append(word)

tagged_sents = CoreNLPPOSTagger().tag(tokens)

t = 0
r = 0
i = 0
for sent in filtered_sents:
    for word,tag in sent:
        if word != tagged_sents[i][0]:
            print(i)
            print(tagged_sents[i][0])
            print(word)
            quit()
        if tag == tagged_sents[i][1]:
            r += 1
        t += 1
        i += 1

print(r/t)



# for sent in sents:
#     i = 0
#     for (word,tag) in sent:
#         if tag == '-NONE-':
#             sent.remove(i)
#
#         i =+ 1

# tokens = []
# tok_sent = []
# for sents in tagged_text:
#     for (word,tag) in sents:
#         tok_sent.append(word)
#     tokens.append(tok_sent.copy())
#     tok_sent.clear()

# tagger = CoreNLPPOSTagger(url='http://localhost:9000')

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