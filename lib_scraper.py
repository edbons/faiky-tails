import requests
import bs4
import re
import os


def tag_text(tag: bs4.element.Tag):
    return [" ".join([string.replace('~', '').replace('--', '-') for string in tag.stripped_strings])]

def postproccess_text(text: str):
    text = re.sub('(\[[\w\s]*\])', "", text)
    return text.strip()


def add_par(input_data: list, output_data: list):
    for tag in input_data:
        if 'примечани' in tag.text.lower(): 
            break
        elif 'КОММЕНТАРИ'.lower() in tag.text.lower(): 
            break
        for child in tag.contents:
            if child.name == 'i':
                continue
            elif 'Типография'.lower() in tag.text.lower():
                continue
            elif 'Оригинал находится здесь:'.lower() in tag.text.lower():
                continue
            elif 'Источник текста:'.lower() in tag.text.lower():
                continue
            elif 'В кн.:'.lower() in tag.text.lower():
                continue
            elif 'OCR'.lower() in tag.text.lower():
                continue
            elif 'Иллюстраци'.lower() in tag.text.lower():
                continue
            elif 'Издание книгопродавца'.lower() in tag.text.lower():
                continue
            elif 'Книга:'.lower() in tag.text.lower():
                continue
            elif 'Собрание сочинений'.lower() in tag.text.lower():
                continue     
            elif 'Набор:'.lower() in tag.text.lower():
                continue
            elif re.findall(' том.?\d+', tag.text.lower()):
                continue                       
            elif child.name == 'div':
                continue
            elif child.name == 'font':
                if len(child.contents) > 0 and child.contents[0].name is not None:
                    continue
                else:    
                    output_data.extend(tag_text(tag))
                    break
            elif isinstance(child, bs4.element.Comment):
                continue
            elif isinstance(child, bs4.element.NavigableString):
                if re.match("([А-Я]{1}\.)", str(child).strip()) is not None:  # skip started initials "В. П. Авенариус -- Детские сказки"
                    continue
                elif re.match('.*[a-zA-Z]{2}.*', str(child).strip()) is not None: # skip strings with 2 or more eng letters  
                    continue
                elif not re.findall('[\w]', str(child).strip()):  # skip strings without words "---------"
                    continue
                elif re.findall('([1-2][0|7-9]\d\d)', str(child).strip()): # skip strings with years, examples "1891" "1992"
                    continue
                elif re.findall(' стр.', str(child).strip()):  # skip strings with pages count - 'Том 3, стр. 289 - 292.'
                    continue
                elif len(str(child).strip()) > 2:
                    output_data.extend([str(child).strip()])
                else:
                    print('Fail child tag', type(child), repr(child))
                    
            else:
                continue
    
def main():
    root_url = 'http://az.lib.ru'
    prefix_tails = '/janr/index_janr_5'
    pages_root_count = 6
    ds_path = 'dataset/raw_other/'
    descr_file_path = os.path.join('dataset', 'description_other.txt')
    filtered_file_path = os.path.join('dataset', 'description_other_filtered.txt')

    rus_lit_urls = ['http://az.lib.ru/rating/litarea/index_1.shtml', 
            'http://az.lib.ru/rating/litarea/index_2.shtml', 
            'http://az.lib.ru/rating/litarea/index_4.shtml', 
            'http://az.lib.ru/rating/litarea/index_3.shtml']

    rus_authors = []
    rus_authors_filtered = ['zelinskij_f_f', 'kun_n_a', 'ershow_p_p', 'remizow_a_m']  # filter greek epic tails and other bad examples

    for url in rus_lit_urls:
        try:
            r = requests.get(url, headers={'Accept-Language': 'ru-RU,ru;q=0.5'})
        except Exception:
            print(f'Page {url} connection errors!')
            break
        else:
            if r:
                payload = bs4.BeautifulSoup(r.content, features='html5lib')                        
                articles_dl = payload.find_all('dl')
                if articles_dl:
                    for part in articles_dl:                
                        for tag in part.children:
                            if tag.name == 'a':
                                author = re.findall("\w{2,}", tag["href"])
                                if author[0] not in rus_authors_filtered:                          
                                    rus_authors.extend(author)


    for file in os.listdir(ds_path):
        os.remove(os.path.join(ds_path, file)) 


    if os.path.exists(descr_file_path):
        os.remove(descr_file_path)
    descr_file = open(descr_file_path, 'a', encoding='utf-8')

    if os.path.exists(filtered_file_path):
        os.remove(filtered_file_path)
    filtered_file = open(filtered_file_path, 'a', encoding='utf-8')

    for i in range(1, pages_root_count + 1):
        try:
            r_url = f'{root_url}{prefix_tails}-{i}.shtml'
            r = requests.get(r_url, headers={'Accept-Language': 'ru-RU,ru;q=0.5'})
        except Exception:
            print(f'Page {r_url} connection errors!')
            break
        else:
            if r:
                payload = bs4.BeautifulSoup(r.content, features='html5lib')                        
                articles_li = payload.find_all('li')
                urls = []
                if articles_li:
                    for part in articles_li:                
                        for tag in part.children:
                            if tag.name == 'a' and 'text' in tag["href"]:                            
                                urls.append(tag["href"])
                
                for url in urls:
                    author = re.findall("\w{2,}", url)[0]
                    if author in rus_authors:
                        r = requests.get(f'{root_url}{url}', headers={'Accept-Language': 'ru-RU,ru;q=0.5'})
                        if r:
                            payload = bs4.BeautifulSoup(r.content, features='html5lib')
                            title = re.sub('[^а-яА-Я\-\s]', '', payload.title.text).replace('Классика', '').strip()
                            articles_dd = payload.find_all('dd')
                            
                            if articles_dd:
                                pars = []
                                add_par(articles_dd, pars)  
                                if pars:
                                    text = '\n'.join(pars)
                                    text = postproccess_text(text)
                                    
                                    if len(text.split()) < 2000:
                                        descr_file.write(f'{title} | {len(text.split())} | {root_url}{url}\n')
                                    
                                        with open(os.path.join(ds_path, f'{title}.txt'), 'w', encoding='utf-8') as f:
                                            f.write(text)
                                    else:
                                        filtered_file.write(f'{title} | {len(text.split())} | {root_url}{url}\n')
                                else:
                                    print(f'Fail, cant find paragraphs: {title} | {root_url}{url}')

    descr_file.close()
    filtered_file.close()

if __name__ == "__main__":
    main()