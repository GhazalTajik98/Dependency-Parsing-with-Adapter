# config.py
import torch


# Dataset settings
DATASET_PATH = "universal-dependencies/universal_dependencies"
BASQUE_DATASET_NAME = "eu_bdt"
ENGLISH_DATASET_NAME = "en_ewt"
DATASET_NAME = ENGLISH_DATASET_NAME

# Project settings
PROJECT_NAME = "TokenDependency"
EXPERIMENT_NAME = "COLI_PROJECT"

# Dependency relations
ALL_DEPRELS = [
    "acl", "acl:relcl", "advcl", "advcl:relcl", "advmod", "advmod:emph", "advmod:lmod", "amod", "appos",
    "aux", "aux:pass", "case", "cc", "cc:preconj", "ccomp", "clf", "compound", "compound:lvc",
    "compound:prt", "compound:redup", "compound:svc", "conj", "cop", "csubj", "csubj:outer",
    "csubj:pass", "dep", "det", "det:numgov", "det:nummod", "det:poss", "discourse", "dislocated",
    "expl", "expl:impers", "expl:pass", "expl:pv", "fixed", "flat", "flat:foreign", "flat:name",
    "goeswith", "iobj", "list", "mark", "nmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubj:outer",
    "nsubj:pass", "nummod", "nummod:gov", "obj", "obl", "obl:agent", "obl:arg", "obl:lmod",
    "obl:tmod", "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp",
    "det:predet", "obl:npmod", "nmod:npmod"
]

# Model and training hyperparameters
HIDDEN_DIM = 768
OUTPUT_DIM = 256
RELATION_NUM = len(ALL_DEPRELS)
SKIP_INDEX = -100
EPOCHS = 15
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
BASE_MODEL_NAME = "xlm-roberta-base"
SAVED_MODEL_NAME = "base_model.pth"


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'