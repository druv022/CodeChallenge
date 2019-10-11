import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from  preprocess import TextPreprocessor
import pickle

stop_words = set(stopwords.words('english'))

def read(file_path: str):
    """Generator function to read JSON file
    
    Arguments:
        file_path {str} -- path to JSON file
    """
    with open(file_path, 'r') as f:
        for i in f.readlines():
            yield(i)

def get_desc(file_path):
    """Generator function to return 'description' from JSON file
    
    Arguments:
        file_path {[type]} -- path to JSON file
    """
    cleaner = TextPreprocessor()
    with open(file_path, 'r') as f:
        for i in f.readlines():
            j_data = json.loads(i)
            text = cleaner.pre_process(j_data['description'])
            word_tokens = [word_tokenize(i) for i in sent_tokenize(text)]
            for sent in word_tokens:
                filtered_sent = sent
                # [filtered_sent.append(i) for i in sent if i not in stop_words]
                yield(filtered_sent)

def get_docs(file_path):
    """ return document dump for temporary usage. DO NOT USE!
    """
    with open(file_path,'rb') as f:
        doc_id = pickle.load(f)

    return doc_id

if __name__ == '__main__':
    data = read('/media/druv022/Data2/Challenge/data/job_descriptions.json/Temp.json')
    # data = read('/media/druv022/Data2/Challenge/data/job_descriptions.json/all_en_descriptions.json')

    for i, item in enumerate(data):
        j_data = json.loads(item)
        print(j_data['company'])

        if i == 100:
            break
