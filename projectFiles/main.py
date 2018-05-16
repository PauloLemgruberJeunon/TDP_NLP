import projectFiles.matrixutils as mu
import projectFiles.utils as utils
import projectFiles.constants as cts
from projectFiles.Utils import xlsxUtils

import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import distance

la = np.linalg


def load_from_txt(path_dict, input_name, encoding, save_in_xlsx, workbook_name, enable_verb_filter,
                  enable_lemmatization, chapter='NA'):

    path_to_input, path_to_output_txt, path_to_output_xlsx, path_to_output_img = get_chapter_or_normal_paths(chapter,
                                                                                                             path_dict)

    verb_filter_lemmatization_code = get_verb_filter_lemmatization_code(enable_lemmatization=enable_lemmatization,
                                           enable_verb_filter=enable_verb_filter)

    feature_name = get_correct_files_feature_identification(chapter, verb_filter_lemmatization_code)

    text_string_input = utils.read_text_input(path_to_input + input_name, encoding, True)
    tokens_list = utils.tokenize_string(text_string_input, True)
    tagged_tokens = utils.tag_tokens_using_stanford_corenlp(tokens_list)
    # Will create context windows (word windows). A list that contains lists of tagged words
    content_dict = utils.tokens_to_centralized_windows(tagged_tokens, 15, enable_verb_filter)

    utils.save_verb_frequency(content_dict['verb_frequency_dict'], path_to_output_txt, feature_name, encoding)

    if enable_verb_filter:
        utils.create_cognitive_level_distribution_graph(path_to_output_img,
                                                        'cognitive_level_frequency_chapter' + chapter + '.pdf',
                                                        content_dict['verb_frequency_dict'])

    quit()

    windows = content_dict['windows']
    # Will create the co-occurrence matrix with the obtained windows
    cooc_matrix = mu.CoocMatrix(windows, enable_lemmatization=enable_lemmatization)


    # U, s, Vh = la.svd(cooc_matrix.matrix.transpose(), full_matrices=True)
    #
    # max = -10
    # sim_matrix = np.zeros((42,42))
    # for i in range(len(cooc_matrix.verb_columns)):
    #     for j in range(len(cooc_matrix.verb_columns)):
    #         sim_matrix[i][j] = distance.euclidean((U[i,0], U[i, 1]), (U[j,0], U[j,1]))
    #         if sim_matrix[i][j] > max:
    #             max = sim_matrix[i][j]
    #
    # sim_matrix = np.divide(sim_matrix, max)
    #
    # workbook = xlsxUtils.MyExcelFileWrite()
    # simwb = utils.create_workbook('/home/paulojeunon/Documents/simMatrixTest.xlsx')
    # sh = utils.get_new_worksheet('test', simwb)
    # utils.write_matrix_in_xlsx(utils.invert_dictionary(cooc_matrix.verb_columns), utils.invert_dictionary(cooc_matrix.verb_columns), sim_matrix, sh)
    # utils.close_workbook(simwb)
    #
    # matrix_size = len(content_dict['verb_frequency_dict'])
    # contextual_verb_sim_matrix = np.zeros([matrix_size, matrix_size])
    # final_list_of_verbs = []
    # matrix_columns_rows_index = {}
    # for level_name in cts.names_of_cognitive_levels:
    #     final_list_of_verbs += cts.cognitive_levels[level_name + '_verbs']
    #
    #
    # k = 0
    # max_number = len(final_list_of_verbs)
    # while k < max_number:
    #     if final_list_of_verbs[k] not in content_dict['verb_frequency_dict']:
    #         del final_list_of_verbs[k]
    #         max_number -= 1
    #         k -= 1
    #
    #     k += 1
    #
    # i = 0
    # for i_verb in final_list_of_verbs:
    #     matrix_columns_rows_index[i_verb] = i
    #
    #     j = 0
    #     for j_verb in final_list_of_verbs:
    #         contextual_verb_sim_matrix[i][j] = cooc_matrix.cosine_column_sim(i_verb, j_verb, 'to ')
    #         j += 1
    #
    #     i += 1
    #
    # workbook = utils.create_workbook('/home/paulojeunon/Desktop/' + '42_verbs_similarity_contextual(ED)_test.xlsx')
    # worksheet = utils.get_new_worksheet('verbs_sim', workbook)
    #
    # inverted_dict = utils.invert_dictionary(matrix_columns_rows_index)
    #
    # utils.write_matrix_in_xlsx(inverted_dict, inverted_dict, contextual_verb_sim_matrix, worksheet)
    # utils.close_workbook(workbook)
    #
    # matrix_cognitive_levels = {}
    # for level in cts.names_of_cognitive_levels:
    #     matrix_cognitive_levels[level] = {}
    #     for level2 in cts.names_of_cognitive_levels:
    #         matrix_cognitive_levels[level][level2] = 0
    #
    # for verb1 in cooc_matrix.verb_columns.keys():
    #     for verb2 in cooc_matrix.verb_columns.keys():
    #         verb1_level = utils.get_verb_cognitive_level(verb1[3:], False)
    #         verb2_level = utils.get_verb_cognitive_level(verb2[3:], False)
    #
    #         i = cooc_matrix.verb_columns[verb1]
    #         j = cooc_matrix.verb_columns[verb2]
    #
    #         if i == j:
    #             continue
    #
    #         matrix_cognitive_levels[verb1_level][verb2_level] += sim_matrix[i][j]
    #
    # max_value = -10
    # for key in matrix_cognitive_levels.keys():
    #     for key2 in matrix_cognitive_levels.keys():
    #         matrix_cognitive_levels[key][key2] = round(matrix_cognitive_levels[key][key2] * 100 / (42))
    #         if max_value < matrix_cognitive_levels[key][key2]:
    #             max_value = matrix_cognitive_levels[key][key2]
    #
    # workbook = utils.create_workbook('/home/paulojeunon/Desktop/' + '6_cognitive_levels_similarity_contextual(ED)_test.xlsx')
    # worksheet = utils.get_new_worksheet('levels_sim', workbook)
    #
    # i = 1
    # for key in cts.names_of_cognitive_levels:
    #     worksheet.write(i, 0, key)
    #     i += 1
    #
    # j = 1
    # for key in cts.names_of_cognitive_levels:
    #     worksheet.write(0, j, key)
    #     j += 1
    #
    # i = 1
    # for key in cts.names_of_cognitive_levels:
    #     j = 1
    #     for key2 in cts.names_of_cognitive_levels:
    #         worksheet.write(i, j, matrix_cognitive_levels[key][key2])
    #         j += 1
    #     i += 1
    #
    # utils.close_workbook(workbook)
    #
    # quit()

    utils.save_highest_elements_on_coocmatrix(cooc_matrix, path_to_output_txt, feature_name, encoding)

    utils.save_noun_frequency(cooc_matrix.noun_freq, path_to_output_txt, feature_name, encoding)

    pure_matrix = cooc_matrix.matrix

    # Will calculate the PPMI of the normal and the filtered matrices
    cooc_matrix.matrix = mu.CoocMatrix.calc_ppmi(cooc_matrix.matrix, 1, 1)

    # This will set a variable of the object 'cooc_matrix' aloowing for the creation of the soc_pmi_matrix
    cooc_matrix.is_pmi_calculated = True

    # Will create the Second order co-occurrence matrix from the filtered co-occurrence matrix matrix
    # cooc_matrix.create_soc_pmi_matrix(cooc_matrix.matrix)

    if save_in_xlsx:
        xlsxUtils.save_matrix_in_xlsx(cooc_matrix, pure_matrix, path_to_output_xlsx, workbook_name)

    if chapter == 'NA':
        utils.calculate_generate_sim_matrix_gdf(cooc_matrix, path_dict, mu.calculate_sim_matrix_from_list, 0.55)

    return cooc_matrix.plot_two_word_vectors


def load_from_xlsx(xlsx_file_path):
    content = xlsxUtils.load_from_wb(xlsx_file_path)
    cooc_matrix = mu.CoocMatrix(build_matrix=False, content=content)

    return cooc_matrix.plot_two_word_vectors


def semantic_similarity_interview_graph(path_dict, all_stages=False, eliminate_same_department_edges=True):
    methods = cts.all_semantic_similarity_methods + ['average_of_methods']

    if all_stages:
        stages_dict = xlsxUtils.read_all_stages(cts.data['interview']['path_to_input'],
                                            'all_nouns_for_hypernyms_treated.xlsx', 'list_of_verbs.xlsx')

        for stage in stages_dict.keys():
            curr_stage_dict = stages_dict[stage]
            content_dict = mu.calculate_sim_matrix_from_list(curr_stage_dict["noun_list"], methods)
            unknown_words = content_dict['unknown_words']
            noun_to_noun_sim_matrices = content_dict['noun_to_noun_sim_matrices']
            utils.save_noun_sim_matrix_in_gdf_2(noun_to_noun_sim_matrices, curr_stage_dict["noun_list"],
                                                curr_stage_dict["department_list"],
                                                curr_stage_dict["full_noun_and_verb_list"],
                                                curr_stage_dict["synset_list"], methods,
                                                path_dict['path_to_output_gdf_interview_sim'],
                                                'semanticSimFromXlsx_' + stage, eliminate_same_department_edges)
    else:
        curr_stage_dict = xlsxUtils.read_all_nouns(cts.data['interview']['path_to_input'],
                                            'all_nouns_for_hypernyms_treated.xlsx', 'list_of_verbs.xlsx')
        content_dict = mu.calculate_sim_matrix_from_list(curr_stage_dict["noun_list"], methods)
        noun_to_noun_sim_matrices = content_dict['noun_to_noun_sim_matrices']
        unknown_words = content_dict['unknown_words']
        utils.save_noun_sim_matrix_in_gdf_2(noun_to_noun_sim_matrices, curr_stage_dict["noun_list"],
                                            curr_stage_dict["department_list"],
                                            curr_stage_dict["full_noun_and_verb_list"],
                                            curr_stage_dict["synset_list"], methods,
                                            path_dict['path_to_output_gdf_interview_sim'],
                                            'semanticSimFromXlsx', eliminate_same_department_edges)


def hypernym_interview_graph(all_stages=False):

    if all_stages:
        stages_dict = xlsxUtils.read_all_stages(cts.data['interview']['path_to_input'],
                                            'all_nouns_for_hypernyms_treated.xlsx', 'list_of_verbs.xlsx')
    else:
        content_dict = xlsxUtils.read_all_nouns(cts.data['interview']['path_to_input'],
                                            'all_nouns_for_hypernyms_treated.xlsx', 'list_of_verbs.xlsx')
        stages_dict = {'all nouns': content_dict}

    utils.execute_java(stages_dict, all_stages)


def load_from_chapters(path_dict, encoding, save_in_xlsx, enable_verb_filter, enable_lemmatization):

    if enable_lemmatization:
        verb_filter_lemmatization_code = '_1'
    else:
        verb_filter_lemmatization_code = '_0'

    if enable_verb_filter:
        verb_filter_lemmatization_code += '_1'
    else:
        verb_filter_lemmatization_code += '_0'

    number_of_chapters = int(path_dict['number_of_chapters']) + 1

    for i in range(1, number_of_chapters):
        load_from_txt(path_dict, 'chapter' + str(i), encoding, save_in_xlsx, 'chapter' + str(i) +
                      verb_filter_lemmatization_code + '.xlsx', enable_verb_filter, enable_lemmatization, str(i))


def create_verb_co_occurrence_matrix():
    import nltk
    import heapq

    lemmatizer = nltk.stem.WordNetLemmatizer()

    text_string_input = utils.read_text_input(cts.data['product_design_and_development']['path_to_input'] + 'input.txt',
                                              'utf-8', True)
    sentences = nltk.tokenize.sent_tokenize(text_string_input)

    regexp_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    tokenized_sentences = [regexp_tokenizer.tokenize(sent) for sent in sentences]
    tokenized_sentences = [sent for sent in tokenized_sentences if len(sent) > 5 and len(sent) < 50]

    tagger = nltk.tag.stanford.CoreNLPPOSTagger(url='http://localhost:9000')

    tagged_sentences = [tagger.tag(sent) for sent in tokenized_sentences]
    verbs_in_each_sentence = [[lemmatizer.lemmatize(word, 'v') for word,tag in sent if tag.startswith('V')]
                              for sent in tagged_sentences]
    print(verbs_in_each_sentence)

    v2i_row = {}
    i2v_row = {}

    v2i_column = {}
    i2v_column = {}

    index_row = 0
    index_column = 0
    for sent in verbs_in_each_sentence:
        skip_loop = True
        for verb in sent:
            if verb in cts.verbs_to_keep:
                skip_loop = False
                break

        if skip_loop:
            continue

        for verb in sent:
            if verb in cts.verbs_to_keep:
                if verb not in v2i_row:
                    v2i_row[verb] = index_row
                    i2v_row[index_row] = verb
                    index_row += 1
            else:
                if verb not in v2i_column:
                    v2i_column[verb] = index_column
                    i2v_column[index_column] = verb
                    index_column += 1

    sim_matrix = np.zeros((len(v2i_row),len(v2i_column)), dtype=int)

    for sent in verbs_in_each_sentence:
        if len(sent) > 1:
            for verb1 in sent:
                try:
                    i = v2i_row[verb1]
                except:
                    continue
                for verb2 in sent:
                    try:
                        j = v2i_column[verb2]
                        sim_matrix[i][j] += 1
                    except:
                        continue

    wb = xlsxUtils.MyExcelFileWrite(cts.home + '../', 'verb_co-occurrence_matrix_PDandD_42Xall.xlsx')
    wb.add_new_worksheet('matrix')
    wb.write_matrix_in_xlsx('matrix', sim_matrix, i2v_row, i2v_column)
    wb.close_workbook()

    txt_out = open(cts.home + '../synthesis_fo_verb_co-occurrence_matrix.txt', 'w')
    for level in cts.names_of_cognitive_levels:
        txt_out.write('Cognitive level: ' + level + '\n')

        verbs = cts.cognitive_levels[level + '_verbs']
        for verb in verbs.keys():
            try:
                row_index = v2i_row[verb]
            except:
                continue
            txt_out.write('\t verb: ' + verb + '\n')
            row = sim_matrix[row_index]
            bigger_freq_indices = heapq.nlargest(20, range(len(row)), row.take)
            bigger_column_verbs = [i2v_column[index] for index in bigger_freq_indices]
            bigger_column_freq = [sim_matrix[row_index][j] for j in bigger_freq_indices]
            j = 0
            for column_verb in bigger_column_verbs:
                txt_out.write('\t\t* ' + column_verb + ' - ' + str(bigger_column_freq[j]) + '\n')
                j += 1

        txt_out.write('\n')


    list_row_verbs_ordered = list(zip(*utils.sort_dict(v2i_row)))[0]
    list_of_row_synsets = [cts.verbs_to_keep[verb] for verb in list_row_verbs_ordered]
    list_column_verbs_ordered = list(zip(*utils.sort_dict(v2i_column)))[0]

    unknown_words, matrix = mu.calculate_sim_matrix_from_dif_lists(list_of_row_synsets, list_column_verbs_ordered)

    gdf_out = open(cts.home + '../' + '42VerbXall_graph_PDandD.gdf', 'w')
    gdf_out.write('nodedef>name VARCHAR, frequency INTEGER\n')

    for verb in list_row_verbs_ordered:
        curr_row = sim_matrix[v2i_row[verb]]
        gdf_out.write(verb + ',' + str(np.sum(curr_row)) + '\n')

    for verb in list_column_verbs_ordered:
        if verb in unknown_words:
            continue
        curr_column = sim_matrix[:,v2i_column[verb]]
        gdf_out.write(verb + ',' + str(np.sum(curr_column)) + '\n')

    gdf_out.write('edgedef>node1 VARCHAR, node2 VARCHAR, weight FLOAT\n')

    i = 0
    while i < len(v2i_row):
        j = 0
        while j < len(v2i_column):
            if i2v_column[j] in unknown_words:
                j += 1
                continue

            if matrix[i][j] < 0.4:
                j += 1
                continue

            gdf_out.write(i2v_row[i] + ',' + i2v_column[j] + ',' +
                   '{0:.2f}'.format(matrix[i][j]) + '\n')
            j += 1
        i += 1

    gdf_out.close()


def get_verb_filter_lemmatization_code(enable_lemmatization, enable_verb_filter):

    if enable_lemmatization:
        verb_filter_lemmatization_code = '_1'
    else:
        verb_filter_lemmatization_code = '_0'

    if enable_verb_filter:
        verb_filter_lemmatization_code += '_1'
    else:
        verb_filter_lemmatization_code += '_0'

    return verb_filter_lemmatization_code


def get_chapter_or_normal_paths(chapter, path_dict):

    if chapter == 'NA':
        path_to_input = path_dict['path_to_input']
        path_to_output_txt = path_dict['path_to_output_txt']
        path_to_output_xlsx = path_dict['path_to_output_xlsx']
        path_to_output_img = path_dict['path_to_output_img']
    else:
        path_to_input = path_dict['path_to_input_chapters']
        path_to_output_txt = path_dict['path_to_output_txt_chapters']
        path_to_output_xlsx = path_dict['path_to_output_xlsx_chapters']
        path_to_output_img = path_dict['path_to_output_chapters']

    return [path_to_input, path_to_output_txt, path_to_output_xlsx, path_to_output_img]


def get_correct_files_feature_identification(chapter, verb_filter_lemmatization_code):
    return '_chapter' + chapter + verb_filter_lemmatization_code


# load_from_txt(cts.data['product_design_and_development'], 'filtered_input.txt', 'utf-8', False, 'asdf', True,
#                   True, chapter='NA')

create_verb_co_occurrence_matrix()
# hypernym_interview_graph()
# hypernym_interview_graph(True)
# semantic_similarity_interview_graph(cts.data['interview'], True)
# semantic_similarity_interview_graph(cts.data['interview'], False)
