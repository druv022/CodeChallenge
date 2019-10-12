from read import read
from collections import Counter
from graph import Graph
from utils import Vocabulary
from preprocess import TextPreprocessor
from gensim.models.wrappers import FastText
import json
from tqdm import tqdm
import spacy
import pickle
import os
import fasttext
from datetime import datetime
import argparse
import sys


nlp = spacy.load("en_core_web_sm")

class Filter():
    """This class identifies skills in a given text based on the following heuristics.
        Assumptions:    1) Skills are mostly NOUN or forms of NOUN.
                        2) Skills can be identified as Entity in a given text (We use spacy for NER)
                        3) Given a description, the most common skill/skills (or prominent concepts) can be extracted using word embedding (eucledian) space.
        Approach:       1) Identify POS tags for all the words and keep only NOUN and forms of NOUN
                        2) Identify Named Entity and keep only a set of entities by this rule:
                            a) Select only if belonging to a predefined set.(predefined set selected based on visual inspection.)
                            b) If single word, its POS be NOUN or its other form
                            c) if multiword, atleast one word must be a NOUN or other forms
                        3) Keep a counter of all the neighboring words using word embedding similarity of selected candidates of step 2
                        4) Select top m most frequent neighboring words signifying the common concept within the text (or found entity)
                        5) Evaluate a similarity score between all the top k most frequent words of step 5 and candidate entity based on following:
                            a) if single word, directly evaluate the similarity with top k and add them
                            b) if multiword, evaluate similarity for each word with top k and add them followed by division by number of words in the candidate word.
                        6) Sort the candidate list of step 2 based on score of step 5 and select top k.
                        7) Associate a weight to each node based on predefined set(visual inspection)
    """
    def __init__(self, emb_model, topk=10):
        self._model = emb_model
        self._desc = None
        self._doc = None
        self._candidates = None
        self._counter = Counter()
        self._weight = []
        self._scale = [0.8, 0.7,0.7, 0.7, 0.7, 0.6, 0.6, 0.5, 0.3]
        self._NER_labels = ['PRODUCT','PERSON', 'ORG', 'NORP', 'LANGUAGE', 'GPE', 'FAC','WORK_OF_ART','EVENT']
        self.topk = topk


    def reset(self, doc):
        self._desc = None
        self._doc = nlp(doc)
        self._candidates = None
        self._weight = []
        self._counter = Counter()

    def __filter_POS(self):
        """ Filter only noun
        """
        candidates = []
        for token in self._doc:
            if token.pos_ in ['NOUN','PROPN'] and token.text not in candidates:
                candidates.append(token.text)
        self._candidates = candidates
        return candidates

    def __filter_NER(self):
        """Filter NER based on predefined list(created using visual inspection)
        """
        candidates = []
        for ent in self._doc.ents:
            if ent.label_ in self._NER_labels and ent.text not in candidates:
                # check if multiword entity also noun
                words = ent.text.split()
                flag = False
                for word in words:
                    if word in self._candidates:
                        flag = True
                        break
                if flag:
                    candidates.append(ent.text)
                    self._weight.append(self._scale[self._NER_labels.index(ent.label_)])
        self._candidates = candidates    
        return candidates

    def __eval_neighbor(self):
        """ Evaluate top m(2topk if topk < 10 else topk) most similar neighboring words and store it counter
        """
        for item in self._candidates:
            if len(item.split()) > 1:
                for word in item.split():
                    try:
                        self._counter.update(self._model.most_similar(word.lower(), topn= self.topk*2 if self.topk < 10 else self.topk))
                    except:
                        pass
            else:
                try:
                    self._counter.update(self._model.most_similar(item.lower(), topn=self.topk*2 if self.topk < 10 else self.topk))
                except:
                    pass

    def __eval_similar(self):
        """Sort the candidate list based on similarity score between top count of most similar neighboring words and candidates and return topk
        """
        top_words = self._counter.most_common(10)
        score = []
        for words in self._candidates:
            words_split = words.split()
            if len(words_split) > 1:
                temp_score = []
                for word in words_split:
                    try:
                        temp_score.append(sum([self._model.similarity(word.lower(),i[0][0]) for i in top_words]))
                    except:
                        pass
                score.append(sum(temp_score)/len(words_split))
            else:
                try:
                    score.append(sum([self._model.similarity(words.lower(),i[0][0]) for i in top_words]))
                except:
                    pass

        sorted_candidates = [(x,y) for x,y,_ in sorted(zip(self._candidates, self._weight, score), key=lambda items: items[2], reverse=True)]
        if self.topk < len(sorted_candidates):
            self._candidates = sorted_candidates[0:self.topk]
        else:
            self._candidates = sorted_candidates

    def process(self, text: str) -> list:
        """ Excute the huristics one by one 
        
        Arguments:
            text {str} -- description as string
        
        Returns:
            list -- list of candiate skills
        """
        # begin by reseting previous state
        self.reset(text)
        self._desc = text

        self.__filter_POS()
        self.__filter_NER()
        self.__eval_neighbor()
        self.__eval_similar()

        return self._candidates

def train_fasttext(filepath) -> str:
    """ Train FASTTEXT by first creating a temporary document of description in txt format for processing by the package.
    
    Arguments:
        filepath {[type]} -- path to JSON file
    
    Returns:
        str -- path to the location of the stored model
    """
    data = read(filepath)
    # create temp txt file for writing the description
    temp_file = os.path.join(os.path.dirname(filepath), 'temp_file.txt')
    
    model_folder = os.path.join(os.path.dirname(filepath),'models')
    if not os.path.exists(model_folder):       
        os.makedirs(model_folder)
    model_file = os.path.join(model_folder, 'fasttext_model.bin')

    cleaner = TextPreprocessor()
    with open(temp_file, 'w') as f:
        for item in tqdm(data):
            j_data = json.loads(item)
            text = cleaner.pre_process(j_data['description'])
            f.write(text+'\n')
    
    model = fasttext.train_unsupervised(temp_file, dim=100, epoch=2, thread=12)
    model.save_model(model_file)

    os.remove(temp_file)
    print('Trained model saved at: ', model_file)
    return model_file

def learn_graph(filepath: str, model_path: str, graph_path: str):
    """Learn the graph by first reading each entry in the JSON file, processing it get the probable skills.
    Approach(after skill identification): 1) Add the title as a node to a Graph
                2) Add the sorted skills identified in step 5 as nodes.
                3) Join edges between title and edges based on step 7
                4) Update the edge weight if new weight evaluated for any other doc is more than current weight
    
    Arguments:
        filepath {str} -- path to JSON file
        model_path {str} -- path to word embedding model file in gensim keyvector formate
        graph_path {str} -- path to graph
    """
    data = read(filepath)
    model = FastText.load_fasttext_format(model_path)
    graph = Graph()

    filter = Filter(model)
    for i, d in tqdm(enumerate(data)):
        j_data = json.loads(d)

        title = j_data['title']
        description = j_data['description'].replace('\n',' ')
        reqd_pos = description.find('require')
        desc = description[reqd_pos:] if reqd_pos > 0 else description

        # add graph node
        graph.add_node(title.lower(), 'title')

        skills = filter.process(desc)
        
        for item in skills:
            skill = item[0].strip()
            graph.add_node(skill.lower(),'skill')
            graph.add_edge(title.lower(), skill.lower(), weight=item[1])

        if i % 100 == 0:
            print('DUPMED: ',i)
            with open(graph_path, 'wb') as f:
                pickle.dump(graph, f)

def get_item(node, number, graph, next_n = False):
    node = node.lower()
    if next_n:
        return graph.next_neighbor(node, number)
    else:
        return graph.nearest_neighbor(node, number)

def main(args):
    if not args.modelpath:
        model_path = train_fasttext(args.filepath)
    else:
        model_path = args.modelpath

    if args.train:
        learn_graph(args.filepath, model_path, args.graphpath)

    with open(args.graphpath, 'rb') as f:
        graph = pickle.load(f)

    if args.t == args.neighbor:
        item = get_item(args.name, args.n, graph, next_n = True)
    else:
        item = get_item(args.name, args.n, graph, next_n = False)

    for i in item:
        print(i)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='Start training from scratch!')
    parser.add_argument('--filepath', type=str, help='Enter full path to JSON file')
    parser.add_argument('--modelpath', type=str, help='Enter path of pre-trained word embedding model in gensim keywordvector format' )

    parser.add_argument('--t', type=str, choices=['skill', 'title'], help='Is it skill or title ? Please specify: ', default='skill')
    parser.add_argument('--name', type=str, help='Name any skill/title', default='ASP.NET')
    parser.add_argument('--neighbor', type=str, choices=['skill','title'],
                        help='What do you want to identify? skill or title', default='skill')
    parser.add_argument('--n', type=int, help='How many skill/title do you want ?', default=5)
    parser.add_argument('--graphpath', type=str, help='Enter trained path:')

    return parser.parse_args(argv)


if __name__ == "__main__":    
    # filepath = '/media/druv022/Data2/Challenge/data/job_descriptions.json/all_en_descriptions.json'
    # # filepath = '/media/druv022/Data2/Challenge/data/job_descriptions.json/Temp.json'
    # modelpath = '/media/druv022/Data2/Challenge/models/fasttext_model.bin'
    # t = 'skill'
    # name = 'ASP.NET'
    # # name = 'Software Developer'
    # neighbor = 'title'
    # n = '5'
    # graphpath = '/media/druv022/Data2/Challenge/data/graph_temp.pkl'

    # sys.argv += ['--filepath', filepath, '--modelpath', modelpath, '--t' , t, '--name', name, '--neighbor', neighbor,'--n',n, '--graphpath', graphpath]

    # SAMPLE FOR COMMAND LINE:
    # python methods.py --filepath /media/druv022/Data2/Challenge/data/job_descriptions.json/all_en_descriptions.json --modelpath /media/druv022/Data2/Challenge/models/fasttext_model.bin --t skill --name ASP.NET --neighbor title --n 5 --graphpath /media/druv022/Data2/Challenge/data/graph_temp.pkl 
    
    main(parse_arguments(sys.argv[1:]))

