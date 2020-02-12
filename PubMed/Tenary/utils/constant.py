
"""
Define constants for Nary task.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PRP': 2, 'NN': 3, 'VBZ': 4, '``': 5, '-LRB-': 6, 'JJR': 7, 'LS': 8, 'MD': 9, 'JJ': 10, 'PRP$': 11, 'RB': 12, "''": 13, 'RP': 14, 'VBP': 15, 'IN': 16, 'CD': 17, 'EX': 18, 'WP$': 19, 'NNS': 20, 'WP': 21, 'VBN': 22, 'PDT': 23, ',': 24, '.': 25, 'NNPS': 26, 'JJS': 27, 'NNP': 28, 'TO': 29, 'WDT': 30, 'SYM': 31, 'POS': 32, 'VBG': 33, 'RBS': 34, 'FW': 35, 'VB': 36, 'RBR': 37, 'CC': 38, 'WRB': 39, 'DT': 40, 'VBD': 41, '-RRB-': 42, ':': 43}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'mwe': 2, 'xcomp': 3, 'next': 4, 'prepc': 5, 'amod': 6, 'appos': 7, 'iobj': 8, 'cop': 9, 'hyphen': 10, 'pcomp': 11, 'preconj': 12, 'partmod': 13, 'prep': 14, 'dep': 15, 'csubjpass': 16, 'neg': 17, 'purpcl': 18, 'tmod': 19, 'poss': 20, 'cc': 21, 'complm': 22, 'infmod': 23, 'agent': 24, 'number': 25, 'acomp': 26, 'abbrev': 27, 'punct': 28, 'rel': 29, 'npadvmod': 30, 'possessive': 31, 'det': 32, 'nsubjpass': 33, 'prt': 34, 'nn': 35, 'quantmod': 36, 'advmod': 37, 'aux': 38, 'ccomp': 39, 'pobj': 40, 'parataxis': 41, 'nsubj': 42, 'expl': 43, 'dobj': 44, 'advcl': 45, 'auxpass': 46, 'predet': 47, 'num': 48, 'conj': 49, 'rcmod': 50, 'csubj': 51, 'self': 52}

DIRECTIONS = 1

NEGATIVE_LABEL = 'None'

LABEL_TO_ID = {'None': 0, 'resistance or non-response': 1, 'sensitivity': 2, 'response': 3, 'resistance': 4}
#LABEL_TO_ID = {'No': 0, 'Yes': 1}

INFINITY_NUMBER = 1e12
