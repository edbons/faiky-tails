import requests
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm
import time

url = 'http://hyaenidae.narod.ru/'
pages_root_prefix = 'story'
pages_root_count = 5
ds_path = 'dataset/raw/'
descr_file_path = 'dataset/description.txt'


def scrape_text(url: str) -> str:
    try:
        req = requests.get(url, headers={'Accept-Language': 'ru-RU,ru;q=0.5'})        
    except Exception:
        print(f'Page {url} connection problem!')
    if req:
        payload = BeautifulSoup(req.content, features='html.parser')         
        
        text = payload.find_all(id='AR')
        if text:
            return text[0].text
        
        text = payload.find_all(id='PAR')
        if text:
            return text[0].text
        
        print(f'Find nothing on {url}')
        return ""
    else:
        print(f"The URL returned {req.status_code} reason {req.reason}!")


def write_text(file_name: str, text: str):
    file_name = re.sub('[?.]', '', file_name)
    path = ds_path + file_name + '.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


if os.path.exists(descr_file_path):
    os.remove(descr_file_path)
descr_file = open(descr_file_path, 'a', encoding='utf-8')

articles = []
for i in tqdm(range(1, pages_root_count + 1)):
    print(f'Proccess story{i}')
    curr_time = time.time()
    try:
        r_url = url + pages_root_prefix + str(i)
        r = requests.get(r_url, headers={'Accept-Language': 'ru-RU,ru;q=0.5'})
    except Exception:
        print(f'Page {r_url} connection errors!')
        break
    else:
        if r:
            payload = BeautifulSoup(r.content, features='html.parser')            
            
            articles_left = payload.find_all(class_='rsleft')
            if articles_left:
                for tag in articles_left[0].children:
                    if tag.name == 'a':
                        article_title = tag.string.rstrip(";")
                        text = scrape_text(tag["href"])
                        if not text:                        
                            print(f'{article_title}, {tag["href"]}, {len(text)}')
                        articles.append((article_title, tag['href'], text))
                        write_text(article_title, text)
                        descr_file.write(f'{article_title}|{tag["href"]}\n')
            
            articles_right = payload.find_all(class_='rsright')
            if articles_right:
                for tag in articles_right[0].children:
                    if tag.name == 'a':
                        article_title = tag.string.rstrip(";")
                        text = scrape_text(tag["href"])
                        if not text:                        
                            print(f'{article_title}, {tag["href"]}, {len(text)}')
                        articles.append((article_title, tag['href'], text))
                        write_text(article_title, text)
                        descr_file.write(f'{article_title}|{tag["href"]}\n')

            articles_liupp = payload.find_all(class_='liupp')
            if articles_liupp:
                for part in articles_liupp:
                    for tag in part.children:
                        if tag.name == 'a':
                            article_title = tag.string.rstrip(";")
                            text = scrape_text(tag["href"])
                            if not text:
                                print(f'{article_title}, {tag["href"]}, {len(text)}')
                            articles.append((article_title, tag['href'], text))
                            write_text(article_title, text)
                            descr_file.write(f'{article_title}|{tag["href"]}\n')

            articles_all = payload.find_all(class_='all')
            if articles_all:
                for tag in articles_all:
                    article_title = tag.string.rstrip(";")
                    text = scrape_text(tag["href"])
                    if not text:                    
                        print(f'{article_title}, {tag["href"]}, {len(text)}')
                    articles.append((article_title, tag['href'], text))
                    write_text(article_title, text)
                    descr_file.write(f'{article_title}|{tag["href"]}\n')
            
            articles_sk = payload.find_all(class_='sk')
            start_count = 270
            if articles_sk:
                for part in articles_sk:                
                    for tag in part.children:
                        if tag.name == 'a':                            
                            article_title = str(start_count) + '. ' + tag.string.rstrip(';')
                            text = scrape_text(tag["href"])
                            if not text:
                                print(f'{article_title}, {tag["href"]}, {len(text)}')
                            articles.append((article_title, tag['href'], text))
                            write_text(article_title, text)
                            descr_file.write(f'{article_title}|{tag["href"]}\n')
                            start_count += 1            
        else:
            print(f"The URL returned {r.status_code}!")
    print(f'Time: {time.time() - curr_time:.2f} s')
    print()
print(f"All documents count: {len(articles)}, last document number: {articles[-1][0].split('.')[0]}")
print(f'Saved files count: {len(os.listdir(ds_path))}')

descr_file.close()