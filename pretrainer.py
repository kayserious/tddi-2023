

from simpletransformers.language_modeling import LanguageModelingModel
import multiprocessing
import glob 
import os
import shutil

class KAYSERIOUSPreTrainer:


    def __init__(self,
                 corpus_path,
                 base_model,
                 pretrain_args,
                 seed,
                 gpu,
                 out_dir = 'continual_pretrained'):
                 
        self.gpu = gpu
                 
        self.out_dir = out_dir
    
        self.pre_model = LanguageModelingModel('bert',base_model,use_cuda = self.gpu)
    
        self.pretrain_args = pretrain_args
        
        self.corpus_path = corpus_path
        
        self.pretrain_args['manual_seed'] = seed
        
        self.pretrain_args['output_dir'] = out_dir
        
    
    def pretrain(self):
    
        self.pre_model.train_model(self.corpus_path,args = self.pretrain_args)
        
        checkpoints_pattern = self.out_dir + '/check*'
    
        for f in glob.glob(checkpoints_pattern):
            shutil.rmtree(f)