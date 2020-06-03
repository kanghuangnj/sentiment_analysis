import fasttext
import os
import numpy as np

pj = os.path.join
root = '/home/ubuntu/sentiment_analysis'
train_prefix = pj(root, 'cv/train')
test_prefix = pj(root, 'cv/test')
word_path = pj(root, 'bert-sentiment/pretrain/word-vectors/crawl-300d-2M.vec')

model_results = []
for category in ['', '_mix_aug1', '_mix_aug2']:
    train_dir = train_prefix + category
    test_dir = test_prefix
    result = []
    for i in range(5):
        model = fasttext.train_supervised(pj(train_dir, str(i)+'.txt'), lr=0.5, epoch=30, dim=300, wordNgrams=3, pretrainedVectors=word_path)
        N, p, r = model.test(pj(test_dir, str(i)+'.txt'))
        print ('folder %d finished ...' % i )
        result.append((p, r))
    model_results.append(result)

for res in model_results:
    print (res)
