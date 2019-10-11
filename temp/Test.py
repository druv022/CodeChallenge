# FILE USED FOR TESTING IDEAS ! DO NOT USE IT

import spacy
from read import read, get_desc, get_docs
import json
from utils import Vocabulary, SentencesIterator
from preprocess import get_entity, get_pos, TextPreprocessor
import pickle
from time import time
from tqdm import tqdm
from gensim.models import Word2Vec
import fasttext
from gensim.models.wrappers import FastText

nlp = spacy.load("en_core_web_sm")
path = '/media/druv022/Data2/Challenge/data/job_descriptions.json/Temp.json'
# path = '/media/druv022/Data2/Challenge/data/job_descriptions.json/all_en_descriptions.json'
data = read(path)

#------------------------------------------------------------------------------------------------
# doc_id = {}
# vocab = Vocabulary()
# for i, item in tqdm(enumerate(data)):
#     j_data = json.loads(item)

#     doc_id[j_data['id']] = [j_data['title'],j_data['description']]

#     vocab.add_document(j_data['description'])
# vocab.build()

# file0 = '/media/druv022/Data2/Challenge/data/read_data.pkl'
# with open(file0,'wb+') as f:
#     pickle.dump(doc_id, f)

# file1 = '/media/druv022/Data2/Challenge/data/vocab.pkl'
# with open(file1,'wb+') as f:
#     pickle.dump(vocab, f)
#-------------------------------------------------------------------------------------------------
# file0= '/media/druv022/Data2/Challenge/data/read_data.pkl'
# doc_id = get_docs(file0)

# cleaner = TextPreprocessor()
# with open('/media/druv022/Data2/Challenge/data/read_data.txt', 'w') as f:
#     for key in tqdm(doc_id):
#         text = doc_id[key][1]
#         text = cleaner.pre_process(text)
#         f.write(text+'\n')
# file1 = '/media/druv022/Data2/Challenge/data/vocab.pkl'
# with open(file1,'rb') as f:
#     vocab = pickle.load(f)

#-----------------------------------------------------------------------
# # desc = SentencesIterator(get_desc(path))
# for i,key in enumerate(desc):
#     print(key)
#     if i > 50:
#         break

# model = Word2Vec(desc, size=100, window=5, min_count=5, workers=12, sg=1, hs=0)
# model.save('Word2Vec.model')

# model = Word2Vec.load('Word2Vec.model')
# print('Here')

#--------------------------------------------------------

# model=fasttext.train_unsupervised('/media/druv022/Data2/Challenge/data/read_data.txt', dim=100, epoch=1, thread=8)
# model.save_model("/media/druv022/Data2/Challenge/data/fasttext_model.bin")

# model = fasttext.load_model("/media/druv022/Data2/Challenge/data/fasttext_model.bin")
# model = FastText.load_fasttext_format("/media/druv022/Data2/Challenge/data/fasttext_model.bin")
# print('Here')

#---------------------------------------------------------
# NER_words = {}
# POS_words = {}

# for key in tqdm(doc_id):
#     title, description = doc_id[key]

#     reqd_pos = description.find('require')
#     desc = j_data['description'][reqd_pos:] if reqd_pos else [:]
#     doc = nlp()
#     NER_words = get_entity(doc, NER_words)
#     POS_words = get_pos(doc, POS_words)

# file2 = '/media/druv022/Data2/Challenge/data/NER_words.pkl'
# with open(file2,'wb+') as f:
#     pickle.dump(NER_words, f)

# file3 = '/media/druv022/Data2/Challenge/data/POS_words.pkl'
# with open(file3,'wb+') as f:
#     pickle.dump(POS_words, f)

#-----------------------------------------------------------

# cleaner = TextPreprocessor()

# for key in doc_id:
#     old_text = doc_id[key][1]
#     clean_text = cleaner.pre_process(old_text)

#     print('Here')

#-----------------------------------------------------------

# with open('/media/druv022/Data2/Challenge/data/vik_data.txt', 'w') as f:
#     for i, item in tqdm(enumerate(data)):
#         j_data = json.loads(item)
#         text = j_data['description']
#         f.write('<START>'+text+'<END>')
