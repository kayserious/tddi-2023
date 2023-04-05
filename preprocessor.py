import pandas as pd
import numpy as np
import string
from zemberek import TurkishSpellChecker,TurkishSentenceNormalizer,TurkishSentenceExtractor,TurkishMorphology,TurkishTokenizer

class KAYSERIOUSPreproc:
    
    def __init__(self,data,text_column):
        
        self.data = data.copy()
        self.text_column = text_column
        self.morphology = TurkishMorphology.create_with_defaults()
        self.normalizer = TurkishSentenceNormalizer(self.morphology)
        self.special_chars = string.punctuation + '\n'
        
    def rm_special_chars(self):
        
        def one_line_rm_special_chars(sentence):

            clean_text = ''.join(char for char in sentence if char not in self.special_chars)

            return clean_text
        
        self.data[self.text_column] = self.data[self.text_column].apply(one_line_rm_special_chars)
        
        
    def normalize_sentences(self):
        
        def one_line_normalize(sentence):

            if len(sentence.split()) != 0:
                return self.normalizer.normalize(sentence)
            else:
                return sentence
            
        self.data[self.text_column] = self.data[self.text_column].apply(one_line_normalize)
        
        
    def stem(self):
    
        def one_line_stem(sentence):
    
            if len(sentence.split()) != 0:

                analysis = self.morphology.analyze_sentence(sentence)

                after = self.morphology.disambiguate(sentence, analysis)

                analysis = after.best_analysis()

                pos = []

                for i, part in enumerate(analysis, start=1):
                    pos.append(part.get_stem())

                return ' '.join(pos)
            else:
                return sentence
            
        self.data[self.text_column] = self.data[self.text_column].apply(one_line_stem)
        
    def rm_stopwords(self):
        
        def one_line_rm_stopwords(sentence):
    
            return ' '.join([word for word in sentence.split() if word not in (get_turkish_sw())])
    
        self.data[self.text_column] = self.data[self.text_column].apply(one_line_rm_stopwords)



        