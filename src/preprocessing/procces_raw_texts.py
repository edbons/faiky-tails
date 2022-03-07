import os
import re
import argparse
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

import spacy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from rake_nltk import Rake, Metric
from tqdm import tqdm

   
def preprocess_texts(output_path: str, file_names: list, name: str, top_kw: int, use_ner: bool=False):
    
    def preprocess_text(text: str=""):
        text = re.sub('\.\.\.', '.', text)
        text = re.sub('—', '-', text).replace('--','-')
        text = re.sub('[«»\"]', ' ', text)
        text = re.sub('\u2003', ' ', text)
        # text = re.sub("\t", " ", text)
        return text.strip()
    
    def extract_context(text: str="", top_kw=20):
        try:            
            rake.extract_keywords_from_text(text)
            top_features = rake.get_ranked_phrases()

            if len(top_features) > top_kw:
                top_features = top_features[:top_kw]
            
            return top_features

        except Exception as e:
            print("Fail Rake on text:", text)
            print("Exception:", e)
    
    rake = Rake(language='russian', 
                stopwords=stopwords.words('russian'), 
                ranking_metric=Metric.WORD_DEGREE, 
                max_length=5, 
                include_repeated_phrases=False)
    if use_ner:
        nlp = spacy.load('ru_core_news_lg')  # TODO: disable unused steps https://spacy.io/usage/processing-pipelines
    
    output_file = os.path.join(output_path, name)
    fout = open(output_file, 'a', encoding='utf-8')

    sentences_to_write = []
    sentences_to_write.append("[KEYWORDS]|[TEXT]\n")

    for file in tqdm(file_names):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            text = preprocess_text(text)                
            context = extract_context(text, top_kw=top_kw)
            if use_ner:
                doc = nlp(text)
                ners = set([ent.text for ent in doc.ents if ent.label_ == 'PER'])
                if len(ners) > 0:
                    for token in doc:
                        if token.text in ners and token.pos_ not in ['NOUN', 'PROPN']:        
                            ners.remove(token.text)     
                    if len(list(ners)) > 10:
                        ners = list(ners)[:10]
                    context = list([word.strip() for word in ners]) + context
            
            context = [re.sub('[^А-Яа-я\s]', '', item).replace("!", "").replace("\n", " ").strip() for item in context]
            context = " _kw_ ".join(context)
            if len(text) < 2:
                print(text)  
            new_sentence =  context + '|' + text.replace('\n','[EOP]') + '\n'
            sentences_to_write.append(new_sentence)
    
    fout.writelines(sentences_to_write) 
    fout.close()
        

def main(args):

    output_path = args.output_path
    
    postfix = ""
    if args.use_ner:
        postfix = "_ner"    
    
    text_files = [os.path.join('dataset/raw', name) for name in os.listdir('dataset/raw')] + \
         [os.path.join('dataset/raw_other', name) for name in os.listdir('dataset/raw_other')]
    
    # 80/10/10
    train, testval = train_test_split(text_files, test_size=0.2)
    val, test = train_test_split(testval, test_size=0.5)

    # TO-DO remove  files fix
    for name in  ['train', 'val', 'test']:    
        try:
            os.remove(os.path.join(output_path, name + postfix))        
        except OSError as error:
            pass
    
    preprocess_texts(output_path, train, 'train' + postfix, args.top_kw, use_ner=args.use_ner)
    preprocess_texts(output_path, val, 'val' + postfix, args.top_kw, use_ner=args.use_ner)
    preprocess_texts(output_path, test, 'test' + postfix, args.top_kw, use_ner=args.use_ner)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--use_discourse', action='store_true', help='use discourse tokens as extra input')
    parser.add_argument('--top_kw', type=int, default=20, help='number of keywords phrases')
    parser.add_argument('--output_path', type=str, default='dataset/full/', help='folder with output texts')
    parser.add_argument('--use_ner', action='store_true', help='Use dataset with NER promt')

    args = parser.parse_args()    
    print(args)
    main(args)    
