import os
import json

with open('../paths.json') as paths_input:
    data = json.load(paths_input)
    print(data)

# Global dictionary used to quickly check if a verb in the text is one that we want to keep in the co-occurrence matrix.
# The values do not matter, only the keys
verbs_to_keep = {'prepare': 1, 'synthesize': 1, 'generate': 1, 'define': 1, 'illustrate': 1, 'classify': 1,
                'develop': 1, 'name': 1, 'defend': 1, 'explain': 1, 'describe': 1, 'criticize': 1,
                'test': 1, 'review': 1, 'order': 1, 'analyze': 1, 'choose': 1, 'create': 1, 'combine': 1, 'infer': 1,
                'extend': 1, 'modify': 1, 'compare': 1, 'indicate': 1, 'distinguish': 1, 'interpret': 1, 'justify': 1,
                'identify': 1, 'list': 1, 'evaluate': 1, 'calculate': 1, 'design': 1, 'recognize': 1, 'model': 1,
                'discuss': 1, 'practice': 1, 'apply': 1, 'estimate': 1, 'compute': 1, 'solve': 1, 'conclude': 1,
                'predict': 1}

knowledge_verbs = {'name': 1, 'list': 1, 'identify': 1, 'describe': 1, 'define': 1, 'recognize': 1, 'order': 1}
comprehension_verbs = {'classify': 1, 'discuss': 1, 'distinguish': 1, 'estimate': 1, 'extend': 1, 'indicate': 1,
                       'review': 1}
application_verbs = {'apply': 1, 'choose': 1, 'compute': 1, 'illustrate': 1, 'modify': 1, 'practice': 1, 'solve': 1}
analysis_verbs = {'analyse': 1, 'calculate': 1, 'compare': 1, 'criticize': 1, 'infer': 1, 'model': 1, 'test': 1}
synthesis_verbs = {'combine': 1, 'create': 1, 'design': 1, 'develop': 1, 'generate': 1, 'prepare': 1, 'synthesize': 1}
evaluation_verbs = {'conclude': 1, 'defend': 1, 'evaluate': 1, 'explain': 1, 'justify': 1, 'interpret': 1, 'predict': 1}


department_colors = {'Chemical': '#000080', 'Civil': '#ff0000', 'Computational': '#228b22', 'Electrical': '#ffff00',
                     'Materials': '#ff1493', 'Mechanical': '#8b4513', 'Mining': '#ffa500', 'Petroleum': '#778899'}

all_semantic_similarity_methods = ['lin', 'jcn', 'wup', 'lch']

sep = os.sep

home = '/home/paulojeunon/Desktop/TDP_NLP/'

path_to_xlsxFolder = home + 'xlsxFiles' + sep
path_to_txtFolder = home + 'txtFiles' + sep
path_to_gdfFolder = home + 'gdfFiles' + sep

path_to_generated_xlsx = path_to_xlsxFolder + 'generatedXlsxFiles' + sep
path_to_interview_xlsx = path_to_xlsxFolder + 'interviewXlsxFiles' + sep

path_to_mec_txt_out = path_to_txtFolder + 'product_design_and_development' + sep
path_to_mec_xlsx_out = path_to_generated_xlsx + 'product_design_and_development' + sep

path_to_interview_sim_gdf = path_to_gdfFolder + 'interviewSimFiles' + sep
path_to_interview_hypernym_gdf = path_to_gdfFolder + 'interviewHypernymFiles' + sep
path_to_book_sim_gdf = path_to_gdfFolder + 'bookSimFiles' + sep

path_to_desktop = home + '..' + sep


