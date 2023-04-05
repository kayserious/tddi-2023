import os
import multiprocessing

import constants
import modeler
import pretrainer

import pandas as pd
import numpy as np


if __name__ == '__main__':

    multiprocessing.freeze_support()


    dp = os.path.join(constants.DATA_PATH,constants.DATA_NAME)

    cp = os.path.join(constants.DATA_PATH,constants.PRETRAIN_DATA_NAME)

    df = pd.read_csv(dp,sep=constants.SEPARATOR)


    df = df[df[constants.TEXT_NAME].str.len() != 1]



    pt = pretrainer.KAYSERIOUSPreTrainer(corpus_path = cp,
                                         base_model = constants.BASE_MODEL,
                                         pretrain_args=constants.PRETRAIN_ARGS,
                                         seed=constants.RANDOM_SEED,
                                         gpu = constants.USE_GPU,
                                         out_dir = constants.SAVE_PRETRAINED_TO)

    pt.pretrain()

    """

    md = modeler.KAYSERIOUSModel(modelargs = constants.MODEL_ARGS,
                                 modelfolder = constants.SAVE_DEPLOYED_TO,
                                 seed = constants.RANDOM_SEED,
                                 gpu = constants.USE_GPU,
                                 base_model = constants.SAVE_PRETRAINED_TO) 
                                
    md.construct_data(training_data = df,text_column = constants.TEXT_NAME,target_column = constants.TARGET_NAME)
                                 
                                 
    md.train_model()
    """