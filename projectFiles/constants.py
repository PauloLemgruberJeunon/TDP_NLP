import os
import json

# Global dictionary used to quickly check if a verb in the text is one that we want to keep in the co-occurrence matrix.
# The values do not matter, only the keys
verbs_to_keep = {'prepare': 'organize.v.05', 'synthesize': 'synthesize.v.01', 'generate': 'generate.v.01',
                 'define': 'define.v.02', 'illustrate': 'exemplify.v.02', 'classify': 'classify.v.01',
                 'develop': 'develop.v.01', 'name': 'name.v.02', 'defend': 'defend.v.01', 'explain': 'explain.v.01',
                 'describe': 'describe.v.02', 'criticize': 'knock.v.06', 'test': 'test.v.01', 'review': 'review.v.01',
                 'order': 'arrange.v.07', 'analyze': 'analyze.v.01', 'choose': 'choose.v.01', 'create': 'create.v.02',
                 'combine': 'combine.v.05', 'infer': 'deduce.v.01', 'extend': 'extend.v.04', 'modify': 'change.v.01',
                 'compare': 'compare.v.01', 'indicate': 'indicate.v.03', 'distinguish': 'distinguish.v.01',
                 'interpret': 'interpret.v.01', 'justify': 'justify.v.01', 'identify': 'identify.v.01',
                 'list': 'list.v.01', 'evaluate': 'measure.v.04', 'calculate': 'calculate.v.02',
                 'design': 'design.v.02', 'recognize': 'recognize.v.02', 'model': 'model.v.05',
                 'discuss': 'hash_out.v.01', 'practice': 'practice.v.01', 'apply': 'use.v.01',
                 'estimate': 'estimate.v.01', 'compute': 'calculate.v.01', 'solve': 'solve.v.01',
                 'conclude': 'reason.v.01', 'predict': 'predict.v.01'}

knowledge_verbs = {'name': 1, 'list': 1, 'identify': 1, 'describe': 1, 'define': 1, 'recognize': 1, 'order': 1}
comprehension_verbs = {'classify': 1, 'discuss': 1, 'distinguish': 1, 'estimate': 1, 'extend': 1, 'indicate': 1,
                       'review': 1}
application_verbs = {'apply': 1, 'choose': 1, 'compute': 1, 'illustrate': 1, 'modify': 1, 'practice': 1, 'solve': 1}
analysis_verbs = {'analyze': 1, 'calculate': 1, 'compare': 1, 'criticize': 1, 'infer': 1, 'model': 1, 'test': 1}
synthesis_verbs = {'combine': 1, 'create': 1, 'design': 1, 'develop': 1, 'generate': 1, 'prepare': 1, 'synthesize': 1}
evaluation_verbs = {'conclude': 1, 'defend': 1, 'evaluate': 1, 'explain': 1, 'justify': 1, 'interpret': 1, 'predict': 1}

cognitive_levels = {'application_verbs': application_verbs, 'knowledge_verbs': knowledge_verbs,
                    'comprehension_verbs': comprehension_verbs, 'analysis_verbs': analysis_verbs,
                    'synthesis_verbs': synthesis_verbs, 'evaluation_verbs': evaluation_verbs}

names_of_cognitive_levels = ['knowledge', 'comprehension', 'application', 'analysis', 'synthesis', 'evaluation']

department_colors = {'Chemical': '#000080', 'Civil': '#ff0000', 'Computational': '#228b22', 'Electrical': '#ffff00',
                     'Materials': '#ff1493', 'Mechanical': '#8b4513', 'Mining': '#ffa500', 'Petroleum': '#778899',
                     'Mechanical2': '#b37700'}

all_semantic_similarity_methods = ['lin', 'jcn', 'wup', 'lch']

sep = os.sep

print(os.environ.get('TDP_NLP_HOME'))

home = '/home/paulojeunon/Desktop/TDP_NLP' + sep

with open(home + 'projectFiles' + sep + 'paths.json') as paths_input:
    data = json.load(paths_input)

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


