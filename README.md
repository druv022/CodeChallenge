# CODE CHALLENGE

This is a reposiory for a code challenge.

### Assumptions:    
1) Skills are mostly NOUN or forms of NOUN.
                        
2) Skills can be identified as Entity in a given text (We use spacy for NER)
                        
3) Given a description, the most common skill/skills (or prominent concepts) can be extracted using word embedding (eucledian) space.
        
### Approach:       
1) Identify POS tags for all the words and keep only NOUN and forms of NOUN
                        
2) Identify Named Entity and keep only a set of entities by this rule:
                            
        a) Select only if belonging to a predefined set.(predefined set selected based on visual inspection.)
                            
        b) If single word, its POS be NOUN or its other form
        c) if multiword, atleast one word must be a NOUN or other forms
                        
3) Keep a counter of all the neighboring words using word embedding similarity of selected candidates of step 2

4) Select top k most frequent neighboring words signifying the common concept within the text (or found entity)

5) Evaluate a similarity score between all the top k most frequent words of step 5 and candidate entity based on following:
        
        a) if single word, directly evaluate the similarity with top k and add them
        b) if multiword, evaluate similarity for each word with top k and add them followed by division by number of words in the candidate word.
                        
6) Sort the candidate list of step 2 based on score of step 5.

7) Associate a weight to each node based on predefined set(visual inspection)

8) Add the title as a node to a Graph
                
9) Add the sorted skills identified in step 6 as nodes.
                
10) Join edges between title and edges based on step 7
                
11) Update the edge weight if new weight evaluated for any other doc is more than current weight

### Execute

To execute

    python methods.py --filepath XXX/Challenge/data/job_descriptions.json/all_en_descriptions.json --modelpath XXX/Challenge/models/fasttext_model.bin --t skill --name ASP.NET --neighbor title --n 5 --graphpath /media/druv022/Data2/Challenge/data/graph_temp.pkl
