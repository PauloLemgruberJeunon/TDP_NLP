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
    from nltk.corpus import wordnet as wn

    # stanford_server = utils.StanfordProcess(cts.home + 'systemScripts/runStanfordCoreNLP.sh')
    # stanford_server.start_process()

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

        filtered_sent = []
        for verb in sent:
            synset_list = wn.synsets(verb, pos='v')
            if synset_list:
               filtered_sent.append(verb)

        for verb in filtered_sent:
            if verb in cts.verbs_to_keep:
                if verb not in v2i_column:
                    v2i_column[verb] = index_column
                    i2v_column[index_row] = verb
                    index_column += 1
            else:
                if verb not in v2i_row:
                    v2i_row[verb] = index_row
                    i2v_row[index_row] = verb
                    index_row += 1


    v2i_column_temp = {}
    new_index = 0
    for cog_level in cts.names_of_cognitive_levels:
        print(cog_level)
        for verb in cts.cognitive_levels[cog_level + '_verbs']:
            if verb in v2i_column:
                v2i_column_temp[verb] = new_index
                new_index += 1

    v2i_column = v2i_column_temp
    i2v_column = utils.invert_dictionary(v2i_column)

    sim_matrix = np.zeros((len(v2i_row),len(v2i_column) + 1), dtype=float)

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
                        sim_matrix[i][j] += 1.0
                    except:
                        continue

    v2i_column['row_sum'] = len(i2v_column)
    i2v_column[v2i_column['row_sum']] = 'row_sum'

    for index in v2i_row.values():
        sum_value = np.sum(sim_matrix[index])
        sim_matrix[index] = np.divide(sim_matrix[index], sum_value)
        sim_matrix[index][v2i_column['row_sum']] = sum_value


    sum_column_index = v2i_column['row_sum']
    sum_column = sim_matrix[:,sum_column_index]
    largests_row_indices = heapq.nlargest(len(sum_column), range(len(sum_column)), sum_column.take)
    ordered_row_verbs = [i2v_row[index] for index in largests_row_indices]

    sim_matrix_ordered = np.zeros((len(v2i_row), len(v2i_column)))

    for i in range(sim_matrix_ordered.shape[0]):
        sim_matrix_ordered[i] = sim_matrix[largests_row_indices[i]]


    v2i_row.clear()
    i2v_row.clear()
    for index,verb in enumerate(ordered_row_verbs):
        v2i_row[verb] = index
        i2v_row[index] = verb


    wb = xlsxUtils.MyExcelFileWrite(cts.home + '../', 'verb_co-occurrence_matrix_PDandD_42Xall.xlsx')
    wb.add_new_worksheet('matrix')
    wb.write_matrix_in_xlsx('matrix', sim_matrix_ordered, i2v_row, i2v_column)
    wb.close_workbook()

    cognitive_dist = {}
    for verb, i_index in v2i_row.items():
        if sim_matrix_ordered[i_index][v2i_column['row_sum']] < 5:
            continue
        cognitive_dist[verb] = {}
        for cog_level_name, verbs in cts.cognitive_levels.items():
            true_lvl_name = cog_level_name[:-6]
            for verb_42, j_index in v2i_column.items():
                if verb_42 in verbs:
                    if true_lvl_name in cognitive_dist[verb]:
                        cognitive_dist[verb][true_lvl_name] += sim_matrix_ordered[i_index][j_index]
                    else:
                        cognitive_dist[verb][true_lvl_name] = sim_matrix_ordered[i_index][j_index]


    gdf_out = open(cts.home + '../' + '42VerbXall_graph_PDandD.gdf', 'w')
    gdf_out.write('nodedef>name VARCHAR\n')

    level_nodes = 'knowledge\ncomprehension\napplication\nanalysis\nsynthesis\nevaluation\n'
    gdf_out.write(level_nodes)

    for verb in cognitive_dist.keys():
        gdf_out.write(verb + '\n')

    gdf_out.write('edgedef>node1 VARCHAR, node2 VARCHAR, weight FLOAT\n')

    for verb, cog_level_dict in cognitive_dist.items():
        for lvl_name, value in cog_level_dict.items():
            if value > 0.1:
                gdf_out.write(verb + ',' + lvl_name + ',' + str(value) + '\n')

    gdf_out.close()

    # stanford_server.kill_process()


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

# create_verb_co_occurrence_matrix()
# hypernym_interview_graph()
# hypernym_interview_graph(True)
# semantic_similarity_interview_graph(cts.data['interview'], True)
# semantic_similarity_interview_graph(cts.data['interview'], False)
