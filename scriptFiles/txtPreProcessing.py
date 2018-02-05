import nltk
import enchant


def complete_filter(input_file_name, output_discarded_words=False, encoding='utf8'):
    input_file = open(input_file_name, 'r', encoding=encoding)
    temp_text = input_file.read()
    input_file.close()

    temp_text = unite_hyphenated_words(temp_text)

    temp_text = treat_strange_symbols(temp_text)

    output_dict = only_english_words(temp_text, output_discarded_words)

    for key in output_dict.keys():
        output_file = open('..\\txtFiles\\' + key + '.txt', 'w', encoding=encoding)
        output_file.write(output_dict[key])
        output_file.close()


def only_english_words(raw_text, output_discarded_words):

    output_dict = {'filtered_txt_output': ''}
    if output_discarded_words:
        output_dict['filtered_txt_output_discarded'] = ''

    en_dict = enchant.Dict('en_US')

    tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw_text)

    for token in tokens:
        if en_dict.check(token):
            output_dict['filtered_txt_output'] += token + ' '
        elif output_discarded_words:
            output_dict['filtered_txt_output_discarded'] += token + ' '

    return output_dict


def unite_hyphenated_words(raw_text):
    return raw_text.replace('-\n', '')


def treat_strange_symbols(txt_input):
    txt_input = txt_input.replace('ﬁ', 'fi')
    return txt_input.replace('ﬂ', 'fl')


def main():
    complete_filter('..\\txtFiles\\pdfToTxt.txt', True)


if __name__ == '__main__':
    main()
