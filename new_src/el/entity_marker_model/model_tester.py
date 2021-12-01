from new_src.el.complex_model.candidate_tester import get_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
from new_src.el.cls_model.model import BertForEntityClassification
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer
)

import transformers
transformers.logging.set_verbosity_error()


config = json.load(open('../../../config/config.json'))
outputpath = config['outputpath']

wexea_directory = outputpath



class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def metrics(preds, out_label_ids):
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(preds)):
        prediction = preds[i]
        label = out_label_ids[i]

        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 1:
            fn += 1

    precision = 0.0
    recall = 0.0
    f1 = 0.0
    if tp > 0 or fp > 0:
        precision = tp / (tp + fp)
    if tp > 0 or fn > 0:
        recall = tp / (tp + fn)
    if precision > .0 or recall > .0:
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1, tp, fp, fn



def evaluate(model, loader):
    loss_acc = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    batches = 0.0
    for batch in tqdm(loader, desc="Evaluating"):
        batches += 1.0
        model.eval()

        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        #entity_mask = batch['entity_mask'].to(device)
        entity_mask = None
        adds_redirect = batch['redirects'].to(device)
        adds_prior = batch['priors'].to(device)
        adds_surname = batch['surnames'].to(device)
        labels = batch['labels'].to(device)



        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, adds_redirect=adds_redirect,
                            adds_surname=adds_surname,
                            token_type_ids=token_type_ids, adds_prior=adds_prior,
                            labels=labels)

            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            loss_acc += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)


    loss_acc = loss_acc / batches
    print("loss: %f" % (loss_acc))
    max_preds = np.argmax(preds, axis=1)
    precision, recall, f1, tp, fp, fn = metrics(max_preds, out_label_ids)
    print("precision: %f, recall: %f, f1: %f" % (precision, recall, f1))
    print("tp: %d, fp: %d, fn: %d" % (tp, fp, fn))

    return preds

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
dataset_dev,test_data_dev = get_dataset(wexea_directory,0, 0,tokenizer=tokenizer, type='msnbc')
loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = 'models/model_base_cls/checkpoint-13079/'
model = BertForEntityClassification.from_pretrained(checkpoint)
model.to(device)

title2id = json.load(open(wexea_directory + 'dictionaries/title2Id.json'))
id2title = json.load(open(wexea_directory + 'dictionaries/id2title.json'))

correct = 0
incorrect = 0

correct_found = 0

one_candidate = {'correct': 0, 'incorrect': 0}
zero_candidate = 0
more_candidates = {'correct': 0, 'incorrect': 0}

preds_dev = evaluate(model, loader_dev)

data_length = len(test_data_dev['contexts'])
preds_idx = 0
for i in range(data_length):
    context = test_data_dev['contexts'][i]
    candidates = test_data_dev['candidates'][i]
    id = test_data_dev['ids'][i]

    if len(candidates) == 0:
        incorrect += 1
        zero_candidate += 1
    elif len(candidates) == 1:
        candidate = candidates[0][0]
        if title2id[candidate] == int(id):
            correct += 1
            correct_found += 1
            one_candidate['correct'] += 1
        else:
            incorrect += 1
            one_candidate['incorrect'] += 1
        preds_idx += 1
    else:
        best_candidate_pred = 0.0
        best_candidate = None
        results = []
        for j in range(len(candidates)):
            #candidate_l.append((candidate[0], prior, redirect, surname, abstract))

            candidate = candidates[j][0]

            f_x = np.exp(preds_dev[preds_idx]) / np.sum(np.exp(preds_dev[preds_idx]))
            prediction = f_x[1]
            if prediction > best_candidate_pred:
                best_candidate_pred = prediction
                best_candidate = candidate

            if int(id) == title2id[candidate]:
                correct_found += 1

            results.append((candidate,prediction))

            preds_idx += 1

        if int(id) == title2id[best_candidate]:
            correct += 1
            more_candidates['correct'] += 1
        else:
            incorrect += 1

            results.sort(key=lambda x:x[1],reverse=True)
            line = ''
            for tuple in results:
                pred = "{:.2f}".format(tuple[1])
                line += tuple[0] + ' (' + pred + ')    '
            line = line.strip()

            print(id2title[id])
            print(line)
            print(context)
            print()

            more_candidates['incorrect'] += 1

print("all: " + str(data_length))
print("correct found: " + str(correct_found))
print("correct: " + str(correct))
print("incorrect: " + str(incorrect))
print("more candidates: ")
print(more_candidates)
print("one candidate:")
print(one_candidate)
print("zero candidate: " + str(zero_candidate))
acc = correct / (correct + incorrect)
print("acc: " + str(acc))
