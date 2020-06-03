import numpy as np
import glob
import os
from random import randint
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
pj = os.path.join
input_path = '../cv/train'
output_path = '../cv/train_mix_aug4' #'../data/train_all_aug.txt' #'../cv/train_all'
zendesk_path = '../data/zendesk.txt'
channel_path = '../data/valid_ex_channel_data.txt'

def prepare_aug():
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
    return aug


def read(path):
    data = []
    with open(path, 'r') as reader:
        for line in reader:
            line = line.strip('\n')
    
            label, text = line.split('\t', 1)
            data.append((text, label))
    return data

def aug_cv(aug):
    sep= '\t'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    channel_data =  read(channel_path)

    for cv_path in glob.glob(pj(input_path, '*.txt')):
        cv_data = read(cv_path)
        all_data = cv_data + channel_data
        data = []
        for text, label in all_data: 
            if label == '__label__Positive':    
                k = 10
            elif label == '__label__Neutral':
                k = 2
            else:
                k = 1
            ids = np.random.randint(0, len(aug), size=k)
            data.append(label + sep + text)
            for k in ids:
                data.append(label + sep + aug[k].augment(text))
        # with open(zendesk_file, 'r') as reader:
        #     for line in reader:
        #         data.append(line)
        np.random.shuffle(data)
        train_file = cv_path.replace(input_path, output_path)
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
                k = 10
            elif gold == '__label__Neutral':
                k = 2
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


aug = prepare_aug()
aug_cv(aug)