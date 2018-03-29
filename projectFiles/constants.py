import os

# Global dictionary used to quickly check if a verb in the text is one that we want to keep in the co-occurrence matrix.
# The values do not matter, only the keys
verbs_to_keep = {'prepare': 1, 'synthesize': 1, 'generate': 1, 'define': 1, 'illustrate': 1, 'classify': 1,
                'develop': 1, 'name': 1, 'defend': 1, 'explain': 1, 'describe': 1, 'criticize': 1,
                'test': 1, 'review': 1, 'order': 1, 'analyze': 1, 'choose': 1, 'create': 1, 'combine': 1, 'infer': 1,
                'extend': 1, 'modify': 1, 'compare': 1, 'indicate': 1, 'distinguish': 1, 'interpret': 1, 'justify': 1,
                'identify': 1, 'list': 1, 'evaluate': 1, 'calculate': 1, 'design': 1, 'recognize': 1, 'model': 1,
                'discuss': 1, 'practice': 1, 'apply': 1, 'estimate': 1, 'compute': 1, 'solve': 1, 'conclude': 1,
                'predict': 1}

sep = os.sep

path_to_xlsxFolder = os.getcwd() + sep + os.path.join('..', 'xlsxFiles') + sep
path_to_txtFolder = os.getcwd() + sep + os.path.join('..', 'txtFiles') + sep
path_to_gdfFolder = os.getcwd() + sep + os.path.join('..', 'gdfFiles') + sep

path_to_generated_xlsx = path_to_xlsxFolder + 'generatedXlsxFiles' + sep
path_to_interview_xlsx = path_to_xlsxFolder + 'interviewXlsxFiles' + sep

path_to_interview_sim_gdf = path_to_gdfFolder + 'interviewSimFiles' + sep
path_to_interview_hypernym_gdf = path_to_gdfFolder + 'interviewHypernymFiles' + sep
path_to_book_sim_gdf = path_to_gdfFolder + 'bookSimFiles' + sep



path_to_desktop = sep + 'home' + sep + 'paulojeunon' + sep + 'Desktop' + sep # Avoid

all_semantic_similarity_methods = ('lin', 'jcn', 'wup', 'lch', 'methods_average')
