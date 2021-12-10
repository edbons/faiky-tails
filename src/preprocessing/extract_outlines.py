from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize
import os
import re
from sklearn.model_selection import train_test_split

def sorting(lst):
    # lst2=sorted(lst, key=len)
    lst2 = sorted(lst, key=len)
    return lst2

def trim_body(body):
    paragraphs = body.replace('<p> ', '\n').strip().split('\n')
    body_new = []
    par_length = 1

    for par in paragraphs:
        _par = par
        if _par.endswith(' <s>.'):
            _par = _par[:-5]
        temp_body = _par.replace(' <s>', ' ').replace('  ', ' ').strip()
        sentences = _par.replace(' <s>', '\n').replace('  ', ' ').strip().split('\n')

        if len(paragraphs) == 1:
            s = 0
            first = True

            while len(sentences[s].split(' ')) < 4 or '::act ' in sentences[s].lower() or ' act:' in sentences[s].lower():
                s+=1
                if s == len(sentences):
                    return None
            body_new.append('<o> ' + sentences[s].replace(' <s> ', ' ').strip())
            s+=1

            while s < len(sentences) and len(body_new)< 5:
                body_new.append('<o>')
                curr_len = 0
                while s < len(sentences) and curr_len + len(sentences[s].split(' ')) < 400:
                    if ':act ' in sentences[s].lower() or 'act: ' in sentences[s].lower() :
                        s+=1
                        break

                    if len(sentences[s]) > 10:
                        curr_len += len(sentences[s].replace(' <s> ', ' ').strip().split(' '))
                        body_new[len(body_new) - 1] += " " + sentences[s].replace(' <s> ', ' ').strip()
                        body_new[len(body_new) - 1] = body_new[len(body_new) - 1].strip()
                    s += 1

        else:
            if par_length >5:
                s = 0
                while s < len(sentences) and len(sentences[s]) > 10 and (len(body_new[len(body_new)-1].split(' ')) + len(sentences[s].split(' '))) < 400:
                    if len(sentences[s]) > 10:
                        body_new[len(body_new) - 1] += " " + sentences[s].replace(' <s> ', ' ').strip()
                        body_new[len(body_new) - 1] = body_new[len(body_new) - 1].strip()
                    s+=1
            else:
                if len(temp_body) > 10 and len(temp_body.split(' ')) <= 400:
                    body_new.append(temp_body.replace(' <s>', ' ').replace('  ', ' ').strip())

                elif len(temp_body.split(' ')) >400:
                    curr_len  = 0
                    newstr = ''
                    for sent in sentences:
                        if len(newstr.split(' ')) + len(sent.split(' ')) <= 400:
                            newstr += (' '+ sent).strip()
                        else:
                            break
                    body_new.append(newstr.replace(' <s>', ' ').replace('  ', ' ').strip())

        par_length+=1

    return body_new

def clean_top_features(keywords, top=10):
    keywords = sorting(keywords)
    newkeys = []
    newkeys.append(keywords[len(keywords)-1])
    for i in range(len(keywords)-2,-1,-1):
        if newkeys[len(newkeys)-1].startswith(keywords[i]):
            continue
        newkeys.append(keywords[i])

    if len(newkeys) > top:
        return newkeys[:10]
    return newkeys

def convert_keys_to_str(key_list):
    newstr = key_list[0]
    for k in range(1, len(key_list)):
        if len(key_list[k].split(' ')) > 2 :
            newstr += '[SEP]' + key_list[k]
    return newstr.replace("(M)", "").strip()

def preprocess_raw_texts(path: str, file_names: list, plot_file: str, title_file: str):
    f_plot = open(plot_file, 'a', encoding='utf-8')
    f_title = open(title_file, 'a', encoding='utf-8')
    for name in file_names:
        f_inp = open(path + name, 'r', encoding='utf-8')
        text = " ".join(f_inp.readlines())
        text = text.replace('\n', ' ')
        text = re.split('[\?\.\!\:\;]', text)
        f_plot.writelines(['  <p> ' + sentence.strip() + ' <s> ' for sentence in text])

        f_plot.write("\n<EOS>\n")
        title = re.sub("[^А-Яа-я]" , " ", name.split('.')[0]).strip() +'\n'
        title = re.sub(" +" , " ", title)
        f_title.write(title)
        f_inp.close()

    f_plot.close()
    f_title.close()         

def preprocess_texts(path: str, file_names: list, plot_file: str, title_file: str, output_file: str):
    
    preprocess_raw_texts(path, file_names, plot_file, title_file)

    f = open(plot_file, 'r', encoding='utf-8')
    f_title = open(title_file, 'r', encoding='utf-8')
    fout = open(output_file, 'a', encoding='utf-8')

    lines = f.readlines()
    lines_title = f_title.readlines()

    # abstract_lens = {}

    sentences_to_write = []
    w = 0
    total = 0
    sentences_to_write.append("[ID]\t[KEY/ABSTRACT]\t[KEYWORDS]\t[DISCOURSE (T/I/B/C)]\t[NUM_PARAGRAPHS]\t[PARAGRAPH]\t[PREVIOUS_PARAGRAPH]\n")

    title_id = 0
    for l in range(len(lines)):
        if lines[l].strip().startswith("<EOS>"):
            continue
        title = lines_title[title_id].strip()
        title_id+=1
        document = lines[l].replace('t outline . <s>', '').replace(' <p> ', ' ').replace('  ', ' ').strip().replace(' <s> ', '\n').split('\n')
        body = lines[l].replace('t outline . <s>', '').strip()

        try:
            r = Rake()
            r.extract_keywords_from_sentences(document)
            top_features = r.get_ranked_phrases()
            top_features = clean_top_features(top_features, topK)
        except Exception:
            print(document)
            continue

        keywordsSTR = convert_keys_to_str(top_features)

        if len(title) > 2:
            title = title.lower().replace("paid notice :", "").replace("paid notice:", "").replace("journal;", "").strip()
            keywordsSTR = title + '[SEP]' + keywordsSTR
            if len(keywordsSTR.split(' ')) > 100:
                keywordsSTR = ' '.join(keywordsSTR.split(' ')[0:100]).strip()

        body_new = trim_body(body)

        if body_new is None or len(body_new) < 1 or len((' '.join(body_new)).split(' '))<15:
            continue

        id = 'plot-' + str(title_id)

        total+=1
        new_sentence = id + '_0\tK\t' + keywordsSTR + '\tI\t' + str(len(body_new)) + "\t" + body_new[0] + "\tNA"
        sentences_to_write.append(new_sentence + '\n')

        for d in range(1, len(body_new)-1):
            new_sentence = id + '_' + str(d) + '\tK\t' + keywordsSTR + '\tB\t' + str(len(body_new)) + "\t" + body_new[d] + "\t" + body_new[d-1]
            sentences_to_write.append(new_sentence + '\n')

        if len(body_new) > 1:
            new_sentence = id + '_' + str(len(body_new)-1) + '\tK\t' + keywordsSTR + '\tC\t' + str(len(body_new)) + "\t" + body_new[len(body_new)-1] + "\t" + body_new[len(body_new)-2]
            sentences_to_write.append(new_sentence + '\n')
        else:
            print(id)

    fout.writelines(sentences_to_write)
    print("Total=" + str(total))


r = Rake()
vectorizer = TfidfVectorizer(ngram_range=(1,3))
topK = 10

input_file_path = 'dataset/raw/'
output_path = 'dataset/plot/'

[os.remove(output_path + file) for file in os.listdir(output_path)]

text_files = os.listdir(input_file_path)
trainval, test = train_test_split(text_files, test_size=0.1)
train, val = train_test_split(trainval, test_size=0.2)

preprocess_texts(input_file_path, train, 'dataset/plot/train_plot', 'dataset/plot/train_title', 'dataset/plot/train_encoded.csv')
preprocess_texts(input_file_path, val, 'dataset/plot/val_plot', 'dataset/plot/val_title', 'dataset/plot/val_encoded.csv')
preprocess_texts(input_file_path, test, 'dataset/plot/test_plot', 'dataset/plot/test_title', 'dataset/plot/test_encoded.csv')
