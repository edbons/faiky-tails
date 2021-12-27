from rake_nltk import Rake
import os
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import argparse


def trim_body(body):
    paragraphs = body.replace('<p>', '\n').strip().split('\n')
    body_new = []
    par_length = 1

    for par in paragraphs:
        _par = par
        temp_body = _par.replace('  ', ' ').strip()
        sentences = [sent.strip() for sent in _par.strip().strip('<s>').split('<s>') if len(sent) > 1]

        if par_length == 1:
            s = 0

            while len(sentences[s].split(' ')) < 4:
                s += 1
                if s == len(sentences):
                    return None
            
            body_new.append('<o> ' + sentences[s].replace('<s>', ' ').strip())
            s += 1

            while s < len(sentences) and len(body_new) < 5:
                body_new.append('<o>')
                curr_len = 0
                while s < len(sentences) and curr_len + len(sentences[s].split(' ')) < 400:
                    if ':act ' in sentences[s].lower() or 'act: ' in sentences[s].lower() :
                        s += 1
                        break

                    if len(sentences[s]) > 10:
                        curr_len += len(sentences[s].replace(' <s> ', ' ').strip().split(' '))
                        body_new[len(body_new) - 1] += " " + sentences[s].replace(' <s> ', ' ').strip()
                        body_new[len(body_new) - 1] = body_new[len(body_new) - 1].strip()
                    s += 1

        else:
            if par_length > 5:
                s = 0
                while s < len(sentences) and len(sentences[s]) > 10 and (len(body_new[len(body_new)-1].split(' ')) + len(sentences[s].split(' '))) < 400:
                    if len(sentences[s]) > 10:
                        body_new[len(body_new) - 1] += " " + sentences[s].replace(' <s> ', ' ').strip()
                        body_new[len(body_new) - 1] = body_new[len(body_new) - 1].strip()
                    s+=1
            else:
                if len(temp_body) > 10 and len(temp_body.split(' ')) <= 400:
                    body_new.append(temp_body.replace(' <s>', ' ').replace('  ', ' ').strip())

                elif len(temp_body.split(' ')) > 400:
                    curr_len  = 0
                    newstr = ''
                    for sent in sentences:
                        if len(newstr.split(' ')) + len(sent.split(' ')) <= 400:
                            newstr += (' '+ sent).strip()
                        else:
                            break
                    body_new.append(newstr.replace(' <s>', ' ').replace('  ', ' ').strip())

        par_length += 1

    return body_new

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
        f_inp = open(os.path.join(path, name), 'r', encoding='utf-8')
        text = f_inp.read()
        text = text.replace('\n', ' <p> ')
        # pars = text.strip().split('\n')
        # sentences = []        
        f_plot.writelines([text])

        f_plot.write("\n<EOS>\n")
        title = re.sub("[^А-Яа-я]" , " ", name.split('.')[0]).strip() + '\n'
        title = re.sub(" +" , " ", title)
        f_title.write(title)
        f_inp.close()

    f_plot.close()
    f_title.close()         

def preprocess_texts(path: str, output_path: str, file_names: list, plot_file: str, title_file: str, output_file: str, topK: int):
    
    plot_file = os.path.join(output_path, plot_file)
    title_file = os.path.join(output_path, title_file)
    output_file = os.path.join(output_path, output_file)

    preprocess_raw_texts(path, file_names, plot_file, title_file)

    f = open(plot_file, 'r', encoding='utf-8')
    f_title = open(title_file, 'r', encoding='utf-8')
    fout = open(output_file, 'a', encoding='utf-8')

    lines = f.readlines()
    lines_title = f_title.readlines()

    sentences_to_write = []

    total = 0
    sentences_to_write.append("[ID]\t[KEY/ABSTRACT]\t[KEYWORDS]\t[DISCOURSE (T/I/B/C)]\t[NUM_PARAGRAPHS]\t[PARAGRAPH]\t[PREVIOUS_PARAGRAPH]\n")

    title_id = 0
    for l in range(len(lines)):
        if lines[l].strip().startswith("<EOS>"):
            continue
        title = lines_title[title_id].strip()
        title_id += 1

        text = re.sub("\.{1,3}|!\?|\?!|[\?!;:]", "<s>", lines[l])
        text = re.sub("(<s>){2,3}?", "<s>", text)
        text = re.sub("[«»]", " ", text)
        text = text.replace('\u2003', ' ')
        paragraphs = text.replace(' <p> ', '\n').replace('  ', ' ').strip('\n').strip().split('\n')
        body = text.strip()

        try:
            r = Rake(language='russian', stopwords=stopwords.words())
            sentences = []
            for par in paragraphs:
                sentences.extend([sent.strip() for sent in par.strip().strip('<s>').split('<s>') if len(sent) > 1])
            r.extract_keywords_from_sentences(sentences)

            top_features = r.get_ranked_phrases()

            if len(top_features) > topK:
                top_features = top_features[:topK]

        except Exception:
            print(paragraphs)
            continue

        keywordsSTR = convert_keys_to_str(top_features)

        if len(title) > 2:
            title = title.lower().strip()
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

def main(args):

    topK = args.top_k

    input_file_path = args.input_path
    output_path = args.output_path

    [os.remove(output_path + file) for file in os.listdir(output_path)]

    text_files = os.listdir(input_file_path)
    # 90/5/5
    train, testval = train_test_split(text_files, test_size=0.1)
    val, test = train_test_split(testval, test_size=0.5)

    if args.use_discourse:
        preprocess_texts(input_file_path, output_path, train, 'train_plot', 'train_title', 'train_encoded.csv', topK)
        preprocess_texts(input_file_path, output_path, val, 'val_plot', 'val_title', 'val_encoded.csv', topK)
        preprocess_texts(input_file_path, output_path, test, 'test_plot', 'test_title', 'test_encoded.csv', topK)
    else:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_discourse', action='store_true', help='use discourse tokens as extra input')
    parser.add_argument('--top_k', type=int, default=5, help='number of keywords phrases')
    parser.add_argument('--input_path', type=str, default='dataset/raw/', help='folder with raw texts')
    parser.add_argument('--output_path', type=str, default='dataset/plot/', help='folder with output texts')

    args = parser.parse_args()    
    print(args)
    main(args)    
