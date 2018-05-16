import nltk

lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_word(word, pos='n'):
    return lemmatizer.lemmatize(word, pos)


def strip_list(string_list, strip_char=' '):
    return [str(word).strip(strip_char) for word in string_list]


