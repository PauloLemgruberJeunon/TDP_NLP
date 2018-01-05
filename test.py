import nltk, re, pprint

#doc = open('txtDocs/test.txt')
#raw = doc.read().lower()

path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
raw = open(path, 'rU').read().lower()
raw = raw[0:35000]

tokens = nltk.word_tokenize(raw)

tagger = nltk.tag.StanfordPOSTagger('stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger'
                                    , 'stanford-postagger-2017-06-09/stanford-postagger.jar')
tagged_words = tagger.tag(tokens)

verbs_dict = {}

for (word, tag) in tagged_words:
    if tag[0] == 'V':
        if word in verbs_dict:
            verbs_dict[word] += 1
        else:
            verbs_dict[word] = 1
    else:
        continue

verb_fd = nltk.FreqDist(word for (word, tag) in tagged_words if tag[0] == 'V')
verb_fd.plot(cumulative=False)

print(verbs_dict)
