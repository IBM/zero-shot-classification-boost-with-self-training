# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import html
import os
import re
import urllib.request
import tarfile
import zipfile

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from class_names import DATASET_TO_CLASS_NAME_MAPPING

OUT_DIR = './datasets'
RAW_DIR = os.path.join(OUT_DIR, 'raw')


def get_label_name(dataset_name, csv_label_name):
    if dataset_name in DATASET_TO_CLASS_NAME_MAPPING:
        return DATASET_TO_CLASS_NAME_MAPPING[dataset_name][str(csv_label_name)]
    else:
        return csv_label_name.lower()


def load_20_newsgroup():
    def clean_text(x):
        x = re.sub('#\S+;', '&\g<0>', x)
        x = re.sub('(\w+)\\\(\w+)', '\g<1> \g<2>', x)
        x = x.replace('quot;', '&quot;')
        x = x.replace('amp;', '&amp;')
        x = x.replace('\$', '$')
        x = x.replace("\r\n", " ").replace("\n", " ")
        x = x.strip()
        while x.endswith("\\"):
            x = x[:-1]
        return html.unescape(x)

    dataset_name = "20_newsgroup"
    dataset_out_dir = os.path.join(OUT_DIR, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    newsgroups_train = fetch_20newsgroups(subset='train')
    train_df = pd.DataFrame({"text": newsgroups_train["data"], "label": newsgroups_train["target"]})
    train_df["text"] = train_df["text"].apply(lambda x: clean_text(x))
    train_df["label"] = train_df["label"].apply(lambda x: get_label_name(dataset_name, x))
    train_df.to_csv(os.path.join(dataset_out_dir, "unlabeled.csv"), index=False)
    logging.info(f"20_newsgroup unlabeled file created with {len(train_df)} samples")
    newsgroups_test = fetch_20newsgroups(subset='test')
    test_df = pd.DataFrame({"text": newsgroups_test["data"], "label": newsgroups_test["target"]})
    test_df["text"] = test_df["text"].apply(lambda x: clean_text(x))
    test_df["label"] = test_df["label"].apply(lambda x: get_label_name(dataset_name, x))
    test_df.to_csv(os.path.join(dataset_out_dir, "test.csv"), index=False)

    with open(os.path.join(dataset_out_dir, 'class_names.txt'), 'w') as f:
        f.writelines([class_name+'\n' for class_name in sorted(test_df["label"].unique())])

    logging.info(f"20_newsgroup test file created with {len(test_df)} samples")


def load_ag_news_dbpedia_yahoo():
    def clean_text(x):
        x = re.sub('#\S+;', '&\g<0>', x)
        x = re.sub('(\w+)\\\(\w+)', '\g<1> \g<2>', x)
        x = x.replace('quot;', '&quot;')
        x = x.replace('amp;', '&amp;')
        x = x.replace('\$', '$')
        x = ' '.join(x.split())
        while x.endswith("\\"):
            x = x[:-1]
        return html.unescape(x)

    dataset_to_columns = {'ag_news': ["label", "title", "text"],
                          'dbpedia': ["label", "title", "text"],
                          'yahoo_answers': ['label', 'question_title', 'question_content', 'answer']}

    for dataset, column_names in dataset_to_columns.items():
        logging.info(f'processing {dataset} csv files')
        raw_path = os.path.join(RAW_DIR, dataset, f'{dataset}_csv')
        with open(os.path.join(raw_path, 'classes.txt'), 'r') as f:
            idx_to_class_name = dict(enumerate([get_label_name(dataset, row.strip())
                                                for row in f.readlines()]))

        dataset_out_dir = os.path.join(OUT_DIR, dataset)
        os.makedirs(dataset_out_dir, exist_ok=True)

        for dataset_part in ["train", "test"]:
            part_file = os.path.join(raw_path, f'{dataset_part}.csv')
            part_df = pd.read_csv(part_file, header=None)
            part_df.columns = column_names

            if dataset == 'yahoo_answers':
                part_df = part_df[~part_df['answer'].isna()]
                part_df['text'] = part_df.apply(lambda x:
                                                f"{x['question_title']} {x['question_content']} {x['answer']}", axis=1)
            elif dataset == 'ag_news':
                part_df['text'] = part_df.apply(lambda x: f"{x['title']}. {x['text']}", axis=1)

            part_df = part_df[~part_df['text'].isna()]
            part_df['text'] = part_df['text'].apply(lambda x: clean_text(x))
            part_df['label'] = part_df['label'].apply(lambda x: idx_to_class_name[x - 1])
            if dataset_part == 'test':
                part_df.to_csv(os.path.join(dataset_out_dir, f'test.csv'), index=False)

                with open(os.path.join(dataset_out_dir, 'class_names.txt'), 'w') as f:
                    f.writelines([class_name + '\n' for class_name in sorted(part_df["label"].unique())])
            else:
                part_df.to_csv(os.path.join(dataset_out_dir, 'unlabeled.csv'), index=False)


def load_isear():
    dataset_name = 'isear'
    dataset_out_dir = os.path.join(OUT_DIR, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    logging.info(f'processing {dataset_name} csv files')
    df = pd.read_csv(os.path.join(RAW_DIR, 'isear', 'isear.csv'), sep='|', quotechar='"', on_bad_lines='warn')
    df = df[['SIT', 'Field1']]
    df.columns = ['text', 'label']
    df['text'] = df['text'].apply(lambda x: x.replace('á ', ''))
    df["label"] = df["label"].apply(lambda x: get_label_name(dataset_name, x))

    unlabeled_df, test_df = train_test_split(df, test_size=0.2)
    unlabeled_df.to_csv(os.path.join(dataset_out_dir, 'unlabeled.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_out_dir, 'test.csv'), index=False)

    with open(os.path.join(dataset_out_dir, 'class_names.txt'), 'w') as f:
        f.writelines([class_name+'\n' for class_name in sorted(test_df["label"].unique())])


def load_imdb():
    dataset_name = 'imdb'
    dataset_out_dir = os.path.join(OUT_DIR, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    logging.info(f'processing {dataset_name} csv files')
    raw_dir = os.path.join(RAW_DIR, 'imdb', 'aclImdb')
    train = []
    for label in ['pos', 'neg', 'unsup']:
        for file in os.listdir(os.path.join(raw_dir, 'train', label)):
            train.append({'text': open(os.path.join(raw_dir, 'train', label, file)).read().replace('<br />', ' '),
                          'label': get_label_name(dataset_name, label) if label != 'unsup' else ''})
    test = []
    for label in ['pos', 'neg']:
        for file in os.listdir(os.path.join(raw_dir, 'test', label)):
            test.append({'text': open(os.path.join(raw_dir, 'test', label, file)).read().replace('<br />', ' '),
                         'label': get_label_name(dataset_name, label)})

    unlabeled_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)

    unlabeled_df.to_csv(os.path.join(dataset_out_dir, 'unlabeled.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_out_dir, 'test.csv'), index=False)

    with open(os.path.join(dataset_out_dir, 'class_names.txt'), 'w') as f:
        f.writelines([class_name+'\n' for class_name in sorted(test_df["label"].unique())])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    dataset_to_download_url = \
        {
            'isear': 'https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/master/isear.csv',
            'ag_news': 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
            'dbpedia': 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k&confirm=t',
            'yahoo_answers': 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU&confirm=t',
            'imdb': 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        }

    for dataset, url in dataset_to_download_url.items():
        out_dir = os.path.join(RAW_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)
        logging.info(f'downloading {dataset} raw files')
        extension = '.'.join(url.split(os.sep)[-1].split('.')[1:])
        if len(extension) == 0:
            extension = 'tar.gz'
        target_file = os.path.join(out_dir, f'{dataset}.{extension}')
        urllib.request.urlretrieve(url, target_file)
        if extension == 'tar.gz':
            file = tarfile.open(target_file)
            file.extractall(out_dir)
            file.close()
        elif extension == 'zip':
            with zipfile.ZipFile(target_file, 'r') as zip_ref:
                zip_ref.extractall(out_dir)

    load_20_newsgroup()
    load_ag_news_dbpedia_yahoo()
    load_isear()
    load_imdb()