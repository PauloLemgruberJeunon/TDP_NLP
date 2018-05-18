import projectFiles.constants as cts
import projectFiles.utils as utils
import projectFiles.matrixutils as mu


def get_each_verb_from_gdf_string(only_verb_node_string):
    rest = only_verb_node_string
    dict_of_verbs = {}
    while True:
        verb, sep, rest = rest.partition(',')
        if not verb.startswith(' t'):
            break
        dict_of_verbs[verb.strip()] = 0

    return dict_of_verbs


# books_names_list = utils.get_book_names_in_json(False)
books_names_list = ['product_design_and_development','engineering_design']

all_nouns = {}
k = 0
for book_name in books_names_list:

    addr = cts.data[book_name]['path_to_output_gdf'] + 'simGraph_average_of_methods.gdf'
    file = open(addr, 'r')

    color = cts.department_colors[cts.data[book_name]['department']]

    temp_lines = []
    for line in file:
        temp_lines.append(line)

    for i in range(1, len(temp_lines)):

        if temp_lines[i].startswith('edgedef'):
            break

        noun, sep, verbs = temp_lines[i].partition(',')
        dict_of_verbs = get_each_verb_from_gdf_string(verbs)

        if noun not in all_nouns:
            all_nouns[noun] = dict_of_verbs
            all_nouns[noun]['my_color'] = color
        else:
            all_nouns[noun]['my_color'] = '#000000'

            number_of_verbs = len(all_nouns[noun])-1
            if number_of_verbs < 16:

                list_of_verbs = list(dict_of_verbs.keys())
                j = 0
                added = 0
                while j < len(dict_of_verbs) and 16 - number_of_verbs > added:
                    if list_of_verbs[j] not in all_nouns[noun]:
                        all_nouns[noun][list_of_verbs[j]] = 0
                        added += 1

                    j += 1

    k += 1


all_nouns_list = list(all_nouns.keys())
content_dict = mu.calculate_sim_matrix_from_list(all_nouns_list, cts.all_semantic_similarity_methods)

noun_to_noun_sim_matrices = content_dict['noun_to_noun_sim_matrices']
unknown_words = content_dict['unknown_words']

utils.save_noun_sim_matrix_in_gdf3(noun_to_noun_sim_matrices, unknown_words, all_nouns, all_nouns_list,
                                   cts.all_semantic_similarity_methods + ['average_of_methods'], 'all',
                                   cts.data['engineering_design'], edge_min_value=0.4)