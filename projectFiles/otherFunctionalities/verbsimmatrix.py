import projectFiles.constants as cts
import projectFiles.matrixutils as mu
import projectFiles.utils as utils
import pandas as pd
from pandas import ExcelWriter
from nltk.corpus import wordnet as wn
from pandas import ExcelFile


list_of_blooms_levels = [cts.knowledge_verbs.keys(), cts.comprehension_verbs.keys(), cts.application_verbs.keys(),
                         cts.analysis_verbs.keys(), cts.synthesis_verbs.keys(), cts.evaluation_verbs.keys()]

blooms_verbs = []
for list in list_of_blooms_levels:
    blooms_verbs += list

i = 0
matrix_columns_rows_index = {}
for verb in blooms_verbs:
    matrix_columns_rows_index[verb] = i
    i += 1

print(matrix_columns_rows_index)
print('')

noun_to_noun_sim_matrices = mu.calculate_sim_matrix_from_list(blooms_verbs, cts.all_semantic_similarity_methods, 'v')

avg_sim_matrix = noun_to_noun_sim_matrices['methods_average']

print('matrix_shape = ' + str(avg_sim_matrix.shape[0]) + ' ' + str(avg_sim_matrix.shape[1]))

print('\n' + cts.path_to_generated_xlsx + '42_verbs_similarity.xlsx\n')
workbook = utils.create_workbook(cts.path_to_generated_xlsx + '42_verbs_similarity.xlsx')
worksheet = utils.get_new_worksheet('verbs_sim', workbook)

inverted_dict = utils.invert_dictionary(matrix_columns_rows_index)

utils.write_cooc_matrix(inverted_dict, inverted_dict, avg_sim_matrix, worksheet)
utils.close_workbook(workbook)


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
