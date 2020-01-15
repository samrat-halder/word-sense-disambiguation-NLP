#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Participate in the 4705 lexical substitution competition (optional): YES
# Alias: [Oblivion]
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []
    ls = wn.lemmas(lemma, pos = pos)
    for l in ls :
        s = l.synset()
        lxs = s.lemmas()
        for lx in lxs:
            word = lx.name().split('.')[0]
            if word != lemma and word not in possible_synonyms:
                if '_' in word :
                    possible_synonyms.append(word.replace('_',' '))
                else:
                    possible_synonyms.append(word)
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    lemma = context.lemma
    pos = context.pos
    frq_dict = {}
    ls = wn.lemmas(lemma, pos = pos)
    for l in ls :
        s = l.synset()
        lxs = s.lemmas()
        for lx in lxs:
            word = lx.name().split('.')[0]
            if word != lemma and word not in frq_dict.keys():
                frq_dict[word] = lx.count()
            elif word != lemma and word in frq_dict.keys():
                frq_dict[word] = frq_dict[word] + lx.count()
    syn_word = max(frq_dict, key=frq_dict.get)
    return syn_word # replace for part 2

def wn_simple_lesk_predictor(context):
    lemma = context.lemma
    pos = context.pos
    sentence = context.left_context + context.right_context
    sentence = [w for w in sentence if w not in stop_words]
    sentence = [stemmer.stem(w) for w in sentence]
    sentence = set(sentence)
    ovlp_dict = {}
    ls = wn.lemmas(lemma, pos = pos)
    for l in ls :
        s = l.synset()
        lxs = s.lemmas()
        for lx in lxs:
            ss = lx.synset()
            deftn = word_tokenize(ss.definition())
            exmple = [word_tokenize(eg) for eg in ss.examples()]
            exmple = sum(exmple,[])
            if len(ss.hypernyms()) != 0 :
                for hyp in ss.hypernyms():
                    deftn = deftn + word_tokenize(hyp.definition())
                    for eg in hyp.examples():
                        exmple = exmple + word_tokenize(eg) 
            deftn = [stemmer.stem(w) for w in deftn if w not in stop_words]
            exmple = [stemmer.stem(w) for w in exmple if w not in stop_words]
            lesk_set = set(deftn + exmple)
            ovlp_count = len(lesk_set.intersection(sentence))
            if ovlp_count > 0:
                word = lx.name().split('.')[0]
                if word != lemma and word not in ovlp_dict.keys():
                    ovlp_dict[word] = ovlp_count
                elif word != lemma and word in ovlp_dict.keys():
                    ovlp_dict[word] = ovlp_dict[word] + ovlp_count
    if not ovlp_dict:
        syn_word = wn_frequency_predictor(context)
    else:
        syn_word = max(ovlp_dict, key=ovlp_dict.get)
    return syn_word #replace for part 3        
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context):
        lemma = context.lemma
        pos = context.pos
        synonyms = get_candidates(lemma, pos)
        cosine_dict = {}
        for syn in synonyms:
            try : cosine_dict[syn] = self.model.similarity(lemma, syn)
            except: continue
        syn_word = max(cosine_dict, key = cosine_dict.get)
        return syn_word # replace for part 4

    def predict_nearest_with_context(self, context): 
        lemma = context.lemma
        pos = context.pos
        synonyms = get_candidates(lemma, pos)
        left = [w for w in context.left_context if w not in string.punctuation]
        right = [w for w in context.right_context if w not in string.punctuation]
        sentence = left[-5:] + [context.word_form] + right[0:5]
        sentence = [w for w in sentence if w not in stop_words]
        sentence_v = np.zeros(300)
        for w in sentence:
            try : sentence_v = sentence_v + self.model.wv[w]
            except : continue
        cosine_dict = {}
        for syn in synonyms:
            try :
                syn_vec = self.model.wv[syn]
                cosine_dict[syn] = np.dot(syn_vec,sentence_v) / (np.linalg.norm(syn_vec)*np.linalg.norm(sentence_v))
            except : continue
        syn_word = max(cosine_dict, key = cosine_dict.get)
        return syn_word # replace for part 5

    def my_predict_nearest_with_context(self, context): 
        lemma = context.lemma
        pos = context.pos
        ls = wn.lemmas(lemma, pos = pos)
        synonyms = get_candidates(lemma, pos)
        left = [w for w in context.left_context if w not in string.punctuation]
        right = [w for w in context.right_context if w not in string.punctuation]
        sentence = left[-5:] + [context.word_form] + right[0:5]
        sentence = [w for w in sentence if w not in stop_words or w == context.word_form]
        sentence_v = np.zeros(300)
        count = 0
        w_index = sentence.index(context.word_form)
        for w in sentence:
            try : 
                sentence_v = sentence_v + np.exp(-abs(count - w_index)**2) * self.model.wv[w]
            except : pass
            count += 1
        cosine_dict = {}
        for syn in synonyms:
            try :
                syn_vec = self.model.wv[syn]
                cosine_dict[syn] = np.dot(syn_vec,sentence_v) / (np.linalg.norm(syn_vec)*np.linalg.norm(sentence_v))
            except : continue
        if not cosine_dict:
            syn_word = synonyms[0]
        else:
            syn_word = max(cosine_dict, key = cosine_dict.get)
        return syn_word

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):

        #print(context)  # useful for debugging
        #prediction = smurf_predictor(context) 
        #prediction = wn_frequency_predictor(context) #Part 2
        #prediction = wn_simple_lesk_predictor(context) #Part 3
        #prediction = predictor.predict_nearest(context) #Part 4
        #prediction = predictor.predict_nearest_with_context(context) #Part 5
        prediction = predictor.my_predict_nearest_with_context(context) #Part 6
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
