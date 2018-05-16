from projectFiles import utils
from projectFiles.Utils import xlsxUtils

import pandas as pd
import numpy as np
from nltk.tag.stanford import CoreNLPPOSTagger
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic


brown_ic = wordnet_ic.ic('ic-brown.dat')
tagger = CoreNLPPOSTagger(url='http://localhost:9000')

def calculate_semantic_sim(word1, word2, pos1='n', pos2='n'):
    list_of_synsets1 = wn.synsets(word1, pos=pos1)
    list_of_synsets2 = wn.synsets(word2, pos=pos2)

    if not list_of_synsets1:
        return 0.0001
    if not list_of_synsets2:
        return 0.0001

    s1 = list_of_synsets1[0]
    s2 = list_of_synsets2[0]

    total_value = 0

    value = s1.wup_similarity(s2)
    value = utils.limit_value(value, 0.0001, 1.0)
    total_value += value

    value = s1.jcn_similarity(s2, brown_ic)
    value = utils.limit_value(value, 0.0001, 1.0)
    total_value += value

    value = s1.lin_similarity(s2, brown_ic)
    value = utils.limit_value(value, 0.0001, 1.0)
    total_value += value

    value = s1.lch_similarity(s2, brown_ic)
    value = value/3.6375861597263857
    value = utils.limit_value(value, 0.0001, 1.0)
    total_value += value

    total_value /= 4

    return total_value


def get_pos_tagged_list(tokens_list):
    return tagger.tag(tokens_list)


def calculate_sentence_similarity(sent1, sent2):
    counter = 0
    accumulator = 0
    for word1,pos1 in sent1:
        if pos1.startswith('NN'):
            pos1 = 'n'
        elif pos1.startswith('V'):
            pos1 = 'v'
        else:
            continue

        for word2,pos2 in sent2:
            if pos2.startswith('NN'):
                pos2 = 'n'
            elif pos2.startswith('V'):
                pos2 = 'v'
            else:
                continue

            if pos1 != pos2:
                continue

            counter += 1
            accumulator += calculate_semantic_sim(word1, word2, pos1=pos1, pos2=pos2)

    try:
        avg_value = accumulator/counter
        return avg_value
    except ZeroDivisionError:
        print(sent1)
        print(sent2)
        return 0.0


if __name__ == '__main__':
    file_path = '/home/paulojeunon/Desktop/files/AlyonaTasks/Design stage nouns.xlsx'

    excel_file = pd.ExcelFile(file_path)
    sheet = excel_file.parse('Full list with all names')

    design_stage_dict = {}
    for design_stage,content_list in sheet.items():
        design_stage_dict[design_stage] = content_list.tolist().copy()

    del design_stage_dict['Design Stage']

    stages_sim_matrices_ref = {}
    stages_given_names_counter = {}
    for design_stage, content_list in design_stage_dict.items():

        stages_sim_matrices_ref[design_stage] = {'w2i': {}, 'i2w': {}}
        curr_matrix = stages_sim_matrices_ref[design_stage]
        curr_matrix_index = 0

        stages_given_names_counter[design_stage] = {}

        for given_name in content_list:
            given_name = given_name.strip().lower()
            if given_name not in curr_matrix['w2i']:
                curr_matrix['i2w'][curr_matrix_index] = given_name
                curr_matrix['w2i'][given_name] = curr_matrix_index
                curr_matrix_index += 1
                stages_given_names_counter[design_stage][given_name] = 0

            stages_given_names_counter[design_stage][given_name] += 1

    stage_sim_matrices = {}
    for design_stage in design_stage_dict.keys():
        matrix_size_r_c = len(stages_sim_matrices_ref[design_stage]['w2i'])
        stage_sim_matrices[design_stage] = np.zeros((matrix_size_r_c, matrix_size_r_c + 1))

        curr_sim_matrix = stage_sim_matrices[design_stage]
        curr_sim_matrix_ref = stages_sim_matrices_ref[design_stage]
        for sent1, index1 in curr_sim_matrix_ref['w2i'].items():
            sent1 = get_pos_tagged_list(utils.tokenize_string(sent1, True))
            for sent2, index2 in curr_sim_matrix_ref['w2i'].items():
                sent2 = get_pos_tagged_list(utils.tokenize_string(sent2, True))

                curr_sim_matrix[index1][index2] = calculate_sentence_similarity(sent1=sent1, sent2=sent2)

    my_excel_file = xlsxUtils.MyExcelFileWrite('/home/paulojeunon/Desktop/files/AlyonaTasks/', 'results.xlsx')

    for design_stage in design_stage_dict.keys():
        my_excel_file.add_new_worksheet(design_stage)

        curr_sim_matrix = stage_sim_matrices[design_stage]
        curr_i2w_dict = stages_sim_matrices_ref[design_stage]['i2w']

        j = curr_sim_matrix.shape[1] - 1

        curr_i2w_dict_rows = curr_i2w_dict.copy()
        curr_i2w_dict[j] = 'frequencies'
        curr_i2w_dict_columns = curr_i2w_dict.copy()

        for index,row_name in curr_i2w_dict_rows.items():
            curr_frequency = stages_given_names_counter[design_stage][row_name]
            curr_sim_matrix[index][j] = curr_frequency

        my_excel_file.write_matrix_in_xlsx(design_stage, curr_sim_matrix, curr_i2w_dict_rows, curr_i2w_dict_columns)

    my_excel_file.close_workbook()