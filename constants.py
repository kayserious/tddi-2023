DATA_PATH = 'data'
DATA_NAME = 'teknofest_train_final.csv'
PRETRAIN_DATA_NAME = 'pretrain_corpus.txt'
SEPARATOR = '|'
TARGET_NAME = 'target'
TEXT_NAME = 'text'
N_FOLDS = 5
RANDOM_SEED = 1453

USE_GPU = True

BASE_MODEL = 'base_model/bert-base-turkish-cased'

SAVE_PRETRAINED_TO = 'continual_pretrained'

SAVE_DEPLOYED_TO = 'deployed_model'


PRETRAIN_ARGS = {'n_gpu' : 2,
                 'num_train_epochs"':10,
                 'train_batch_size': 128}


MODEL_ARGS =  {
    'use_early_stopping': True,
    'early_stopping_delta': 0.01,
    'early_stopping_metric_minimize': False,
    'early_stopping_patience': 5,
    'evaluate_during_training_steps': 1000,
    'fp16': False,
    'num_train_epochs':12,
    'train_batch_size': 128,
    'n_gpu':2,
    'overwrite_output_dir':True
}