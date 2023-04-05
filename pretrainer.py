

from simpletransformers.language_modeling import LanguageModelingModel
import multiprocessing
import glob 
import os
import shutil

class KAYSERIOUSPreTrainer:

    """
    
    Pre-train edilmiş bir BERT modelini belirtilecek bir derlem (corpus) ile belirtilecek iterasyon adedince tekrar pre-train etmek için kullanılır.
    
    Parametreler:
    
    corpus_path (str) : Pre-train işleminde kullanılacak olan metin dosyasının yolu. Corpus, her satırda bir cümle olacak şekilde hazırlanmalıdır.
    base_model (str) : Eğitime devam edilecek modelin yolu (pytorch ya da tensorflow binary dosyalarını, tokenizer dosyalarını ve config dosyalarını içermelidir). Model belirtilen yolda bulunamazsa huggingface model repolarında aranacaktır. 
    pretrain_args (dict) : Pre-train işleminde kullanılacak MLM (Masked Language Modeling) parametreleri. constants.py dosyasındaki varsayılan parametrelerin kullanılması önerilir.
    seed (int) : Yeniden üretilebilir sonuçlar için kullanılacak anahtar sayı.
    gpu (bool) : Eğitim işlemi sırasında GPU desteği kullanılıp kullanılmayacağını belirleyen mantıksal değer. 
    out_dir (str) : Eğitim tamamlandıktan sonra yeni modelin kaydedileceği dizin.
    
    """


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
    
    
        """
        
        Gerekli düzenlemeleri yapılmış pre-trainer objesini eğitmek ve çıktılarını kaydetmek için kullanılan fonksiyon.
        
        
        """
    
        self.pre_model.train_model(self.corpus_path,args = self.pretrain_args)
        
        checkpoints_pattern = self.out_dir + '/check*'
    
        for f in glob.glob(checkpoints_pattern):
            shutil.rmtree(f)