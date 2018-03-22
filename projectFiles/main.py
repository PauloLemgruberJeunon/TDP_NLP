from py4j.java_gateway import JavaGateway
import projectFiles.matrixutils as mu
import projectFiles.utils as utils
import projectFiles.constants as cts
import os


def load_from_txt(txt_input_path, encoding, save_in_xlsx, workbook_name, enable_verb_filter, enable_lemmatization):

    text_string_input = utils.read_text_input(txt_input_path, encoding, True)

    tokens_list = utils.tokenize_string(text_string_input, True)

    tagged_tokens = utils.tag_tokens_using_stanford_corenlp(tokens_list)

    # Will create context windows (word windows). A list that contains lists of tagged words
    windows = utils.tokens_to_centralized_windows(tagged_tokens, 15, enable_verb_filter)

    # Will create the co-occurrence matrix with the obtained windows
    cooc_matrix = mu.CoocMatrix(windows, enable_lemmatization=enable_lemmatization)

    if save_in_xlsx:
        # The worksheet will be saved in a excel workbook with the file name and location equal to the string below
        workbook = utils.create_workbook(workbook_name)

        pure_frequency_count_sheet = utils.get_new_worksheet('pure_frequency_count', workbook)

        # Just a test worksheet with the pure frequency numbers of the pairs of verbs and nouns
        utils.write_cooc_matrix(utils.invert_dictionary(cooc_matrix.noun_rows),
                                utils.invert_dictionary(cooc_matrix.verb_columns), cooc_matrix.matrix,
                                pure_frequency_count_sheet)


    # Will calculate the PPMI of the normal and the filtered matrices
    cooc_matrix.matrix = mu.CoocMatrix.calc_ppmi(cooc_matrix.matrix, 1, 1)

    # This will set a variable of the object 'cooc_matrix' aloowing for the creation of the soc_pmi_matrix
    cooc_matrix.is_pmi_calculated = True

    # Will create the Second order co-occurrence matrix from the filtered co-occurrence matrix matrix
    # cooc_matrix.create_soc_pmi_matrix(cooc_matrix.matrix)

    if save_in_xlsx:
        # In the lines below the worksheets will be created and associated to the workbook
        worksheet = utils.get_new_worksheet('cooc_matrix_full', workbook)
        worksheet2 = utils.get_new_worksheet('soc_pmi_matrix', workbook)
        worksheet3 = utils.get_new_worksheet('verb_filtered_arrays', workbook)

        inverted_matrix_noun_rows = utils.invert_dictionary(cooc_matrix.noun_rows)

        utils.write_cooc_matrix(inverted_matrix_noun_rows,
                                utils.invert_dictionary(cooc_matrix.verb_columns), cooc_matrix.matrix,
                                worksheet)

        utils.write_cooc_matrix(inverted_matrix_noun_rows,
                                inverted_matrix_noun_rows, cooc_matrix.soc_pmi_matrix,
                                worksheet2)

        utils.write_verb_filtered_arrays(cooc_matrix.nouns_from_verb_arrays, cooc_matrix.verb_filtered_arrays,
                                         worksheet3, workbook)

        utils.close_workbook(workbook)

    # cooc_matrix.calculate_sim_matrix()
    cooc_matrix.noun_to_noun_sim_matrices = mu.calculate_sim_matrix_from_list(list(cooc_matrix.noun_rows.keys()),
                                                                              cts.all_semantic_similarity_methods)
    utils.save_noun_sim_matrix_in_gdf(cooc_matrix , cooc_matrix.noun_rows, cts.all_semantic_similarity_methods,
                                      cts.path_to_txtFolder, 'PDandD')

    return cooc_matrix.plot_two_word_vectors


def load_from_xlsx(xlsx_file_path):
    content = utils.load_from_wb(xlsx_file_path)
    cooc_matrix = mu.CoocMatrix(build_matrix=False, content=content)

    return cooc_matrix.plot_two_word_vectors


def main():
    content_dict = utils.readAllNouns()
    executeJava(content_dict["noun_list"], content_dict["department_list"], content_dict["full_noun_list"],
                content_dict["synset_list"])

    # noun_to_noun_sim_matrices = mu.calculate_sim_matrix_from_list(content_dict["noun_list"])
    # methods = ('wup', 'lch', 'jcn', 'lin', 'average_similarity')
    # utils.save_noun_sim_matrix_in_gdf_2(noun_to_noun_sim_matrices, content_dict["noun_list"],
    #                                     content_dict["department_list"], methods, os.getcwd() + '\\..\\txtFiles\\',
    #                                     'fromXLSX')


def executeJava(noun_list, department_list, full_noun_list, synset_list):
    gateway = JavaGateway()
    word_container_list  = gateway.jvm.java.util.ArrayList()

    i = 0
    for i in range(len(noun_list)):
        print(i)
        word_countainer = gateway.jvm.WordContainer()
        word_countainer.setFullWord(full_noun_list[i])
        word_countainer.setReducedWord(noun_list[i])
        word_countainer.setSynset(str(synset_list[i]).strip())
        curr_dept = department_list[i]
        color = "#000000"
        if curr_dept == "Chemical":
            color = "#000080"
        elif curr_dept == "Civil":
            color = "#ff0000"
        elif curr_dept == "Computational":
            color = "#228b22"
        elif curr_dept == "Electrical":
            color = "#ffff00"
        elif curr_dept == "Materials":
            color = "#ff1493"
        elif curr_dept == "Mechanical":
            color = "#8b4513"
        elif curr_dept == "Mining":
            color = "#ffa500"
        elif curr_dept == "Petroleum":
            color = "#778899"
        word_countainer.setHexColor(color)

        word_container_list.add(word_countainer)

    WordGraph = gateway.entry_point
    WordGraph.startWordGraph(word_container_list, cts.sep + "home" + cts.sep + "paulojeunon" + cts.sep +
                             "Desktop" + cts.sep + "hyperGraph.gdf")


if __name__ == "__main__":
    main()
