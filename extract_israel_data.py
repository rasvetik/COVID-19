"""
Created on Jan 07 2021
Sveta Raboy

"""
# the open source code for Data in gov.il based on ckan: https://github.com/CIOIL/DataGovIL
import os
import time
import re
import requests
import datetime
from bs4 import BeautifulSoup
import pandas as pd


def heb_month_to_eng(date):
    heb_month = ['ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני', 'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר', 'במאי']
    eng_month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'May']
    dd = date.split(' ')
    t = dd[1] #month
    for k in range(12):
        if (heb_month[k] in t):
            dd[1] = eng_month[k]
            break
    #dd[1] = eng_month[heb_month.index(dd[1])]
    dd = dd[:3]
    return datetime.datetime.strptime(''.join(dd), '%d%B%Y').strftime('%d%m%y')


def extract_israel_data(desirable_formats=None):
    # Step 1 - Request
    # This is a trick of israel data base
    headers = {'User-Agent': 'datagov-external-client'}
    url = 'https://data.gov.il/dataset/covid-19'
    r = requests.get(url, headers=headers)
    # HTML parser
    soup = BeautifulSoup(r.content, features='lxml')

    # Step 2 - Parse
    data_locate_root = soup.find_all('body', class_='gov-dir')[0].get('data-locale-root')
    resources = soup.find_all('li', class_='resource-item')
    file_list = []
    if not desirable_formats:
        desirable_formats = ['CSV', 'XLSX']
    for resource in resources:
        file_format = resource.find_all('span', {'class': 'format-label'})[0].text
        if file_format in desirable_formats:
            name = resource.find_all('a', {'class': 'heading'})[0].get('title')
            description = resource.find_all('p', {'class': 'description'})[0].text.strip().replace('\n', '').replace(',', '')
            update = description.split(':')[1].strip()
            data_id = resource.get('data-id')
            try:
                link = data_locate_root[:-1] + resource.find_all('a', href=re.compile(file_format.swapcase()))[0].get('href')
            except:
                other_format = desirable_formats.copy()
                other_format.remove(file_format)
                link = data_locate_root[:-1] + resource.find_all('a', href=re.compile(other_format[0].swapcase()))[0].get('href')

            try:
                cur_file = requests.get(link, headers=headers)
            except:
                cur_file = requests.get(link, headers=headers)

            current_file_name = link.split('/')[-1].replace('-', '_')  # .replace('xlsx', 'csv')
            print('Loading: ', current_file_name, name)
            parent_folder = os.path.join(os.getcwd(), time.strftime("%d%m%Y"))
            new_name = os.path.join(parent_folder, heb_month_to_eng(update) + '_' + current_file_name)
            if os.path.exists(new_name):
                os.remove(new_name)
            elif not os.path.exists(parent_folder):
                os.makedirs(parent_folder, exist_ok=True)
            with open(new_name, 'wb') as f:
                f.write(cur_file.content)


            file = {
                'name': name,
                'last_update': heb_month_to_eng(update),
                'current_file_name': current_file_name,
                'current_file_path': new_name,
                'link': link,
                'data_id': data_id,
                'description': description
            }
            file_list.append(file)
        else:
            continue

    # Step 3 - Output
    df = pd.DataFrame(file_list)

    # Saving full data
    df.to_csv(os.path.join(os.getcwd(), time.strftime("%d%m%Y"), time.strftime("%d%m%Y") + '_loaded_files.csv'),
              index=False, encoding="ISO-8859-8")

    return df
