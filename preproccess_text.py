import os
import re
from ufal.udpipe import Model, Pipeline
import sys
import wget
from tqdm import tqdm
import pickle
from nltk.corpus import stopwords
from string import punctuation

punctuation+='«»'

def remove_stops(text: str, stop_words: set) -> str:
    """Description: удаление русских стоп слов по словарю NLTK
    
    """
    words = text.lower().split()
    text_without_stops = []
    for word in words:
        if word not in stop_words:
            text_without_stops.append(word)

    return ' '.join(text_without_stops)


def check_stops(lemma: str, stop_words: set) -> str:
    """Description: удаление русских стоп слов по словарю NLTK
    
    """     
    return lemma in stop_words

def num_replace(word):
    """Description: замена цифр в словах на символ 'x'
    
    """

    newtoken = 'x' * len(word)
    return newtoken


def clean_token(token, misc):
    """Description: удаление пробелов внутри токена
    
    """

    out_token = token.strip().replace(' ', '')
    if token == 'Файл' and 'SpaceAfter=No' in misc:
        return None
    return out_token


def clean_lemma(lemma, pos):
    """Description: удаление специальных символов из леммы
    
    """
    
    out_lemma = lemma.strip().replace(' ', '').replace('_', '').lower()
    if '|' in out_lemma or out_lemma.endswith('.jpg') or out_lemma.endswith('.png'):
        return None
    if pos != 'PUNCT':
        if out_lemma.startswith('«') or out_lemma.startswith('»'):
            out_lemma = ''.join(out_lemma[1:])
        if out_lemma.endswith('«') or out_lemma.endswith('»'):
            out_lemma = ''.join(out_lemma[:-1])
        if out_lemma.endswith('!') or out_lemma.endswith('?') or out_lemma.endswith(',') \
                or out_lemma.endswith('.'):
            out_lemma = ''.join(out_lemma[:-1])
    return out_lemma


def clean_text(text: str, keep_punct=False) -> str:
    """Description: удаление специальных символов из строки
    
    """
    if keep_punct:
        for c in punctuation:
            text = text.replace(c,' {} '.format(c))
    else:
        text = re.sub('[«»()!,;:.\s-]', ' ', text)
    return text


def process(pipeline, text='Строка', keep_pos=True, keep_punct=False, stop_words: set=None) -> list:
    """Description: токенизация, получение POS тегов, преобразование слов в форму леммы

    """

    entities = {'PROPN'}
    named = False
    memory = []
    mem_case = None
    mem_number = None
    tagged_propn = []

    processed = pipeline.process(text)
    content = [l for l in processed.split('\n') if not l.startswith('#')]
    tagged = [w.split('\t') for w in content if w]
    # print(tagged)

    for t in tagged:
        if len(t) != 10:
            continue
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        token = clean_token(token, misc)
        lemma = clean_lemma(lemma, pos)
        stop_lemma = check_stops(lemma=lemma, stop_words=stop_words)
        if not lemma or not token:
            continue
        if stop_lemma:
            continue
        if pos in entities:
            if '|' not in feats:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}
            if 'Case' not in morph or 'Number' not in morph:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            if not named:
                named = True
                mem_case = morph['Case']
                mem_number = morph['Number']
            if morph['Case'] == mem_case and morph['Number'] == mem_number:
                memory.append(lemma)
                if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN')
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN')
                tagged_propn.append('%s_%s' % (lemma, pos))
        else:
            if not named:
                if pos == 'NUM' and token.isdigit():  # Заменяем числа на xxxxx той же длины
                    lemma = num_replace(token)
                tagged_propn.append('%s_%s' % (lemma, pos))
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN')
                tagged_propn.append('%s_%s' % (lemma, pos))

    if not keep_punct:
        tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PUNCT']
    if not keep_pos:
        tagged_propn = [word.split('_')[0] for word in tagged_propn]
    return tagged_propn



def tag_ud(text, modelfile='udpipe_syntagrus.model', keep_pos=True, keep_punct=False, stop_words: set=None):
    """Description: предобработка текста с вычислением POS тегов udpipe моделью от Русвекторс
    
    """

    udpipe_model_url = 'https://rusvectores.org/static/models/udpipe_syntagrus.model'
    udpipe_filename = udpipe_model_url.split('/')[-1]

    if not os.path.isfile(modelfile):
        print('UDPipe model not found. Downloading...', file=sys.stderr)
        wget.download(udpipe_model_url)


    model = Model.load(modelfile)
    process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

    text = clean_text(text)

    output = process(process_pipeline, text=text, keep_pos=True, keep_punct=False, stop_words=stop_words)
    return output


if __name__== "__main__":   
    test = 'My (very) long, «sentence». No way! But: some thing good;'
    print(clean_text(test, keep_punct=True)) 
