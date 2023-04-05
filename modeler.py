import os
               
from simpletransformers.classification import ClassificationModel

from datasets import Dataset

import evaluate
import torch
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import f1_score


def f_macro(y_true,y_pred):

    """
    
    Değerlendirme metriği olarak kullanılacak ve sırasıyla gözlem / tahmin değerlerini alan skorlama fonksiyonu.
    
    
    """

    return f1_score(y_true = y_true,y_pred=y_pred,average='macro')


class KAYSERIOUSModel:


    """
    
    BERT Modelini sağlanan veriye göre yapılandırıp eğitmek, hali hazırda eğitilmiş bir model varsa canlıya almak için kullanılır.
    
    Parametreler:
    
    live (bool) : Model canlıya alınacaksa True yapılması gereken mantıksal parametre. Eğer live = True olursa raise_from_binary() işlevi ile model dosyası eşlenmelidir.
    modelargs (dict) : Fine-tune işlemi için gerekli olan argümanları içeren dictionary. constants.py dosyasındaki varsayılan değerlerin kullanılması önerilir.
    modelfolder (str) : Eğitim süresince ve tamamlandıktan sonra model dosyalarının kaydedileceği dizinin yolu.
    seed (int) : Yeniden üretilebilir sonuçlar için kullanılacak anahtar sayı.
    gpu (bool) : Eğitim işlemi sırasında GPU desteği kullanılıp kullanılmayacağını belirleyen mantıksal değer. 
    base_model (str) : Fine-tune edilecek modelin yolu (pytorch ya da tensorflow binary dosyalarını, tokenizer dosyalarını ve config dosyalarını içermelidir).
    
    """
    
    def __init__(self,
                 live = False,
                 modelargs = None,
                 modelfolder = None,
                 seed = None,
                 gpu = False,
                 base_model = "dbmdz/bert-base-turkish-cased"):
        
        
        if not live:         
            self.gpu = gpu
            self.seed = seed
            self.modelargs = modelargs
            self.modelfolder = modelfolder
            self.base_model = base_model
    
    def construct_data(self,training_data,text_column,target_column):
    
    
        """
        
        Veri yapılandırma işlevi, bu işlev sayesinde her farklı problem ve model denemesi için farklı ayarlar yapmaya gerek kalmaz. 
        
        Parametreler :
        
        training_data (pandas.DataFrame) : Eğitim verisini içeren bir pandas veri çerçevesi. İçinde metin sütunu ve tahminlenmek istenen değişkeni bulundurmalıdır.
        text_column (str) : Eğitimde kullanılacak metinleri içeren sütun ismi
        target_column (str) : Eğitimde kullanılacak tahminlenmek istenen etiketleri içeren sütun ismi
        
        """
        
        self.id2label = dict(enumerate(training_data[target_column].unique()))
        
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        self.training_frame = training_data.rename(columns = {text_column : 'text',target_column : 'labels'})
        
        self.training_frame = self.training_frame[['text','labels']]
        
        self.training_frame['labels'].replace(self.label2id,inplace=True)
        
        self.modelargs['manual_seed'] = self.seed
        
        self.bare_model = ClassificationModel('bert',self.base_model,args = self.modelargs,use_cuda = self.gpu,num_labels = len(self.label2id))
        
        self.text_column = text_column
        self.target_column = target_column
        
        
    def train_model(self):
    
        """
        
        Veri yapılandırıldıktan sonra model eğitimini başlatır.
        
        
        """
    
        
        self.bare_model.train_model(self.training_frame,acc = f_macro,output_dir = self.modelfolder)
        
        self.bare_model.model.save_pretrained(save_directory = self.modelfolder)
        
        self.bare_model.tokenizer.save_pretrained(save_directory = self.modelfolder)
        
    def predict(self,new_data):
    
    
        """
        
        Tahmin işlevi, yeni veri üzerinde eğitilen model ile tahmin yapılmasını sağlar.
        
        Parametreler :
        
        new_data (pandas.DataFrame) : Metin sütunu içeren bir pandas veri çerçevesi.
        
        """
        
        new_data = new_data.copy()
    
        new_data = new_data[['id','text']].reset_index(drop=True)
        
        preds = self.bare_model.predict(new_data[self.text_column].tolist())
        
        preds = pd.DataFrame({self.target_column : preds[0]})
        
        preds[self.target_column].replace(self.id2label,inplace=True)
        
        with_preds = pd.concat([new_data,preds],axis = 1)
        
        with_preds['is_offensive'] = np.where(with_preds[self.target_column] == 'OTHER', 0, 1).astype(np.int32)
        
        with_preds = with_preds[['id',self.text_column,'is_offensive',self.target_column]]
        
        return with_preds
        
        
    def raise_from_binary(self,binary_path,text_column,target_column):
    
    
        """
        
        Fine-tune edilen bir modeli canlıya almak için kullanılır.
        
        Parametreler : 
        
        binary_path (str) : Model dosyalarını içeren dizinin yolu.
        text_column (str) : Metin değerlerini içeren sütun ismi.
        target_column (str) : Tahminlenmek istenen hedef değişkenin kaydedileceği sütun ismi.
        
        """
    
        self.bare_model = ClassificationModel('bert',binary_path)
        self.text_column = text_column
        self.target_column = target_column
        
        self.id2label = {0: 'INSULT', 1: 'RACIST', 2: 'SEXIST', 3: 'PROFANITY', 4: 'OTHER'}
