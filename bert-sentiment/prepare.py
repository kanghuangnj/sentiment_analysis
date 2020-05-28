import numpy as np
import glob
import os
from random import randint
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
pj = os.path.join
input_path = '../cv/train'
output_path = '../data/train_all_aug.txt' #'../cv/train_all'
zendesk_file = '../data/zendesk.txt'

# Contextual Word Embeddings Augmenter, Substitute word by contextual word embeddings 
neu_aug = []
neu_aug.append(naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert"))
neu_aug.append(naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="substitute"))
neu_aug.append(naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased', action="substitute"))
neu_aug.append(naw.ContextualWordEmbsAug(
    model_path='roberta-base', action="substitute"))

# Synonym Augmenter, Substitute word by WordNet's synonym 
syn_aug = []
syn_aug.append(naw.SynonymAug(aug_src='wordnet'))
syn_aug.append(naw.SynonymAug(aug_src='ppdb', model_path='/home/ubuntu/sentiment_analysis/bert-sentiment/syn_model/ppdb-2.0-tldr'))

# Antonym Augmenter
ant_aug = []
ant_aug.append(naw.AntonymAug())

# Random Word Augmenter
random_aug = []
random_aug.append(naw.RandomWordAug(action="swap"))
random_aug.append(naw.RandomWordAug())

print ('augmenter initialization finished ...')
aug = []
aug.extend(neu_aug)
aug.extend(syn_aug)
aug.extend(ant_aug)
aug.extend(random_aug)

def aug_cv():
    sep= '\t'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for train_file in glob.glob(pj(input_path, '*.txt')):
        data = []
        with open(train_file, 'r') as reader:
            for line in reader:
                gold, text = line.split(sep, 1)
                data.append(line)
                if gold == '__label__Positive':    
                    k = 5
                elif gold == '__label__Neutral':
                    k = 1
                else:
                    k = 1
                ids = np.random.randint(0, len(aug), size=k)
                for k in ids:
                    data.append(gold + sep + aug[k].augment(text))
        # with open(zendesk_file, 'r') as reader:
        #     for line in reader:
        #         data.append(line)
        np.random.shuffle(data)
        train_file = train_file.replace(input_path, output_path)
        with open(train_file, 'w') as writer:
            for line in data:
                line = line.strip('\n')
                writer.write(line)
                writer.write('\n')

def aug_train():
    sep= '\t'
    train_file = '../data/covid.txt'
    data = []
    with open(train_file, 'r') as reader:
        for line in reader:
            gold, text = line.split(' ', 1)
            data.append(gold + sep + text)
            if gold == '__label__Positive':    
                k = 5
            elif gold == '__label__Neutral':
                k = 1
            else:
                k = 1
            ids = np.random.randint(0, len(aug), size=k)
            for k in ids:
                data.append(gold + sep + aug[k].augment(text))
    # with open(zendesk_file, 'r') as reader:
    #     for line in reader:
    #         data.append(line)
    np.random.shuffle(data)
    with open(output_path, 'w') as writer:
        for line in data:
            line = line.strip('\n')
            writer.write(line)
            writer.write('\n')

aug_train()