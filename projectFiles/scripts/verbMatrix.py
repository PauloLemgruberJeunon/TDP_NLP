import projectFiles.constants as cts
import projectFiles.matrixutils as mu
import projectFiles.utils as utils
from projectFiles.Utils import xlsxUtils

import gensim
import numpy as np


model = gensim.models.KeyedVectors.load_word2vec_format('/home/paulojeunon/Desktop/TDP_NLP/GoogleNews-vectors-'
                                                        'negative300.bin', binary=True)

blooms_verbs = ['list', 'name', 'recognize', 'define', 'identify', 'describe', 'order', 'distinguish', 'indicate',
                'review', 'extend', 'classify', 'estimate', 'discuss', 'modify', 'apply', 'illustrate', 'solve',
                'choose', 'compute', 'practice', 'calculate', 'model', 'infer', 'analyze', 'test', 'compare',
                'criticize', 'combine', 'create', 'design', 'synthesize', 'generate', 'develop', 'prepare', 'justify',
                'explain', 'interpret', 'evaluate', 'defend', 'predict', 'conclude']

blooms_verbs_synsets = ['list.v.01', 'name.v.02', 'recognize.v.02', 'define.v.02', 'name.v.02', 'describe.v.02',
                        'arrange.v.07', 'distinguish.v.01', 'indicate.v.03', 'review.v.01', 'extend.v.04',
                        'classify.v.01', 'estimate.v.01', 'hash_out.v.01', 'change.v.01', 'use.v.01', 'exemplify.v.02',
                        'solve.v.01', 'choose.v.01', 'calculate.v.01', 'practice.v.01', 'calculate.v.01', 'model.v.05',
                        'deduce.v.01', 'analyze.v.01', 'test.v.01', 'compare.v.01', 'knock.v.06', 'combine.v.05',
                        'create.v.02', 'design.v.02', 'synthesize.v.01', 'generate.v.01', 'develop.v.01',
                        'organize.v.05', 'justify.v.01', 'explain.v.01', 'interpret.v.01', 'measure.v.04',
                        'defend.v.01', 'predict.v.01', 'reason.v.01']

i = 0
matrix_columns_rows_index = {}
for verb in blooms_verbs:
    matrix_columns_rows_index[verb] = i
    i += 1

print(matrix_columns_rows_index)
print('')

# content_dict = mu.calculate_sim_matrix_from_list(blooms_verbs_synsets, cts.all_semantic_similarity_methods,
#                                                  'v', True, True)
#
# noun_to_noun_sim_matrices = content_dict['noun_to_noun_sim_matrices']
#
# avg_sim_matrix = noun_to_noun_sim_matrices['average_of_methods']

avg_sim_matrix = np.zeros((42,42))
i = 0
for verb1 in matrix_columns_rows_index.keys():
    j = 0
    for verb2 in matrix_columns_rows_index.keys():
        avg_sim_matrix[i][j] = model.wv.similarity(verb1, verb2)
        j += 1
    i += 1

workbook = xlsxUtils.MyExcelFileWrite('/home/paulojeunon/Desktop/', '42_verbs_similarity_w2v.xlsx')
workbook.add_new_worksheet('verbs_sim')

inverted_dict = utils.invert_dictionary(matrix_columns_rows_index)

workbook.write_matrix_in_xlsx('verbs_sim', avg_sim_matrix, inverted_dict, inverted_dict)
workbook.close_workbook()


matrix_cognitive_levels = {}
for level in cts.names_of_cognitive_levels:
    matrix_cognitive_levels[level] = {}
    for level2 in cts.names_of_cognitive_levels:
        matrix_cognitive_levels[level][level2] = 0

for verb1 in blooms_verbs:
    for verb2 in blooms_verbs:
        verb1_level = utils.get_verb_cognitive_level(verb1, False)
        verb2_level = utils.get_verb_cognitive_level(verb2, False)

        i = matrix_columns_rows_index[verb1]
        j = matrix_columns_rows_index[verb2]

        if i == j:
            continue

        matrix_cognitive_levels[verb1_level][verb2_level] += avg_sim_matrix[i][j]

max_value = -10
for key in matrix_cognitive_levels.keys():
    for key2 in matrix_cognitive_levels.keys():
        matrix_cognitive_levels[key][key2] = round(matrix_cognitive_levels[key][key2] * 100 / (42))
        if max_value < matrix_cognitive_levels[key][key2]:
            max_value = matrix_cognitive_levels[key][key2]

# for key in matrix_cognitive_levels.keys():
#     for key2 in matrix_cognitive_levels.keys():
#         matrix_cognitive_levels[key][key2] = 100 * matrix_cognitive_levels[key][key2] / max_value

workbook = xlsxUtils.MyExcelFileWrite('/home/paulojeunon/Desktop/', '6_cognitive_levels_similarity.xlsx')
workbook.add_new_worksheet('levels_sim')

workbook.write_dict_matrix_in_xlsx('levels_sim', matrix_cognitive_levels, cts.names_of_cognitive_levels,
                                cts.names_of_cognitive_levels)

workbook.close_workbook()



# data_frames = []
# temp_dict = {'verbs': []}
# for verb in blooms_verbs:
#     temp_dict['verbs'].append(verb)
#     for synset in wn.synsets(verb, 'v'):
#         synset_string = str(synset) + ': ' + synset.definition()
#         temp_dict['verbs'].append(synset_string)
#
#     temp_dict['verbs'].append('--------------')
#     temp_dict['verbs'].append(' ')
#
# data_frames.append(pd.DataFrame(temp_dict))
# data_frames.append(pd.DataFrame({' ': []}))
# data_frames.append(pd.DataFrame({'Paulo': []}))
# data_frames.append(pd.DataFrame({'Alyona': []}))
# data_frames.append(pd.DataFrame({'Mehwish': []}))
#
# final_data_frame = pd.concat(data_frames, axis=1)
# print(final_data_frame.head())
#
# writer = ExcelWriter('/home/paulojeunon/Desktop/verbsSynsetMeaning.xlsx', engine='xlsxwriter')
# final_data_frame.to_excel(writer, 'sheet1', index=False)
# writer.save()