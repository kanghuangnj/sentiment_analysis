import os
import argparse
import glob
import pandas as pd
from torch.utils.data import RandomSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from dataset import SentimentDataset
from model import SentimentBERT

pj = os.path.join
BERT_MODEL = 'roberta-base' #'pretrain/roberta-pretrain' #'pretrain/bert-pretrain' #'bert-base-uncased' 
NUM_LABELS = 3  # negative, neutral and positive reviews

parser = argparse.ArgumentParser(prog='script')
parser.add_argument('--train', action="store_true", help="Train new weights")
parser.add_argument('--cv', action="store_true", help="Cross validation")
parser.add_argument('--evaluate', action="store_true", help="Evaluate existing weights")
parser.add_argument('--predict', action="store_true", help="Predict sentiment on given file")
parser.add_argument('--path', default='weights/', type=str, help="Weights path")
parser.add_argument('--train-file', default='../data/train_all_aug.txt',
                    type=str, help="covid train file. One sentence per line.")
parser.add_argument('--test-file', default='../data/covidDownload_unlabel.csv',
                    type=str, help="covid test file. One sentence per line.")
parser.add_argument('--train-dir', default='../cv/train_mix_aug4',
                    type=str, help="covid train file. One sentence per line.")
parser.add_argument('--test-dir', default='../cv/test',
                    type=str, help="covid test file. One sentence per line.")
args = parser.parse_args()


def train(train_file, epochs=20, output_dir="roberta_weights/"):
    config = RobertaConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
    tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    model = RobertaForSequenceClassification.from_pretrained(BERT_MODEL, config=config)
    
    # config = BertConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
    # tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    # model = BertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)

    dt = SentimentDataset(tokenizer)
    dataloader = dt.prepare_dataloader(train_file, sampler=RandomSampler)
    predictor = SentimentBERT()
    predictor.train(tokenizer, dataloader, model, epochs)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate(test_file, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)
    dt = SentimentDataset(predictor.tokenizer)
    dataloader = dt.prepare_dataloader(test_file)
    score = predictor.evaluate(dataloader)
    return score

def predict(test_file, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)
    dt = SentimentDataset(predictor.tokenizer)
    id2label = {0: 'Neutral',
                1: 'Negative',
                2: 'Positive', 
            }
    prediction = []
    probility = []
    covid_df = pd.read_csv(test_file)
    covid_text =  covid_df['clean_Text'].tolist()
    for line in covid_text:
        line = str(line)
        text = line.strip('\n')
        dataloader = dt.prepare_dataloader_from_examples([(text, -1)], sampler=None)   # text and a dummy label
        result = predictor.predict(dataloader)
        pred = id2label[result[0][0]]
        probility.append(result[1][0])
        prediction.append(pred)
    covid_df['roberta_pred'] = prediction
    covid_df['roberta_score'] = probility
    covid_df.to_csv('../data/prediction.csv')

def cross_validation(train_dir, epochs=20, output_dir="cv_weights/"):
    scores = []
    os.makedirs(output_dir, exist_ok=True)
    suffix = '.txt'
    for train_file in glob.glob(pj(train_dir, '*' + suffix)):
        idx = train_file.split('/')[-1].strip(suffix)
        data_type = train_file.split('/')[-2]
        output_path = pj(output_dir, data_type, idx)
        os.makedirs(output_path, exist_ok=True)
        train(train_file, epochs=epochs, output_dir=output_path)
        test_file = '../cv/test/'+idx+'.txt'
        print (idx)
        #print ('train...')
        #evaluate(train_file, model_dir=output_path)
        print ('test...')
        score = evaluate(test_file, model_dir=output_path)
        scores.append(score)
    print (scores)

if __name__ == '__main__':
    if args.train:
        os.makedirs(args.path, exist_ok=True)
        train(args.train_file, epochs=30, output_dir=args.path)
    if args.cv:
        cross_validation(args.train_dir, epochs=30)
    if args.evaluate:
        evaluate(args.test_file, model_dir=args.path)

    if args.predict:
        predict(args.test_file)

    #print(predict("It was truly amazing experience.", model_dir=args.path))
