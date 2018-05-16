import projectFiles.utils as utils
import projectFiles.constants as cts
from projectFiles.Utils import xlsxUtils

utils.setup_environment()

stage_list = ['all nouns', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6']
Y_LABEL = 'Noun Count'
OBJECT_LIST = ['Knowledge', 'Comprehension', 'Application', 'Analysis', 'Synthesis', 'Evaluation']

wanted_hypernyms_dict = {'entity': {}, 'abstraction': {}, 'physical_entity': {}}
for dict in wanted_hypernyms_dict.values():
    for cog_level in OBJECT_LIST:
        dict[cog_level.lower()] = {}
        for i in range(1, len(stage_list)):
            dict[cog_level.lower()][stage_list[i]] = 0

for stage in stage_list:

    content_dict = xlsxUtils.read_verb_frequency_from_hypernym_graph(cts.data['interview']['path_to_output_xlsx_hypernym'],
                                                                 'verbFrequency_' + stage + '.xlsx')

    for (hypernym, value_dict) in content_dict.items():
        level = value_dict['level']
        title = hypernym + ' | ' + 'verb: ' + value_dict['hypernymVerb'] + ' | ' + 'level: ' + level

        text_below = ''
        for father in value_dict['fatherList']:
            text_below += father + ' -> '

        text_below = text_below[:-4]

        cognitive_level_counter = {'knowledge_verbs': 0, 'comprehension_verbs': 0, 'application_verbs': 0,
                                   'analysis_verbs': 0, 'synthesis_verbs': 0, 'evaluation_verbs': 0}

        for verb in value_dict['verbList']:
            for (level_name, level_verbs) in cts.cognitive_levels.items():
                if verb[3:] in level_verbs:
                    cognitive_level_counter[level_name] += 1

        y_value_list = [cognitive_level_counter['knowledge_verbs'], cognitive_level_counter['comprehension_verbs'],
                        cognitive_level_counter['application_verbs'], cognitive_level_counter['analysis_verbs'],
                        cognitive_level_counter['synthesis_verbs'], cognitive_level_counter['evaluation_verbs']]

        if stage != 'all nouns':
            if hypernym in wanted_hypernyms_dict:
                for cog_level in OBJECT_LIST:
                    cog_level_right_key = cog_level.lower()
                    key = cog_level_right_key + '_verbs'
                    wanted_hypernyms_dict[hypernym][cog_level_right_key][stage] = cognitive_level_counter[key]

        utils.create_bar_plot(cts.data['interview']['path_to_output_img_hypernym_' + stage.replace(' ', '')],
                              'verbFrequency_' + hypernym + '.pdf', OBJECT_LIST, y_value_list, title, Y_LABEL,
                                      text_below=text_below)

for (hypernym,freq_dict) in wanted_hypernyms_dict.items():
    for (cog_level, stage_dict) in freq_dict.items():

        y_value_list = []
        for i in range(1, len(stage_list)):
            y_value_list.append(stage_dict[stage_list[i]])

        utils.create_bar_plot(cts.data['interview']['path_to_output_img_hypernym_' + cog_level],
                              hypernym + '_' + cog_level + '_frequency.pdf', stage_list[1:], y_value_list,
                              hypernym, 'Knowledge verbs count')


