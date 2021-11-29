from candidate_tester import get_dataset
import torch
import copy
import json
import pickle
import os.path
from tqdm import tqdm
import numpy as np
from model import BertForEntityClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AdamW, BertTokenizer, BertForNextSentencePrediction
)

import transformers
transformers.logging.set_verbosity_error()


config = json.load(open('../../config/config.json'))
outputpath = config['outputpath']

wexea_directory = outputpath



class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


MAX_SENT_LENGTH = 128
EPOCHS = 10

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
num_added_toks = tokenizer.add_tokens(["<e>", "</e>"])
added_tokens = tokenizer.get_added_vocab()
entity_start_token_id = added_tokens['<e>']
print('We have added', num_added_toks, 'tokens')
model = BertForEntityClassification.from_pretrained('bert-base-cased')
model.resize_token_embeddings(len(tokenizer))


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


def run_test(test_dataset, title2id, preds):
    correct = 0
    incorrect = 0

    correct_found = 0

    one_candidate = {'correct': 0, 'incorrect': 0}
    zero_candidate = 0
    more_candidates = {'correct': 0, 'incorrect': 0}

    data_length = len(test_dataset['contexts'])
    preds_idx = 0
    for i in range(data_length):
        context = test_dataset['contexts'][i]
        candidates = test_dataset['candidates'][i]
        id = test_dataset['ids'][i]

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
            for j in range(len(candidates)):
                #candidate_l.append((candidate[0], prior, redirect, surname, abstract))

                candidate = candidates[j][0]

                '''abstract = candidates[j][4]
                prior = candidates[j][1]
                redirect = candidates[j][2]
                surname = candidates[j][3]

                sentence_a = [context]
                sentence_b = [abstract]
                ps = [prior]
                ss = [surname]
                res = [redirect]

                inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=128, truncation=True,
                                   padding='max_length')
                input_ids = inputs['input_ids']

                b = []
                for i in range(len(input_ids)):
                    # for each mention
                    entity_start_token = -1
                    for j in range(len(input_ids[i])):
                        if input_ids[i][j] == entity_start_token_id:
                            entity_start_token = j
                            break

                    entity_mask = [False] * len(input_ids[i])
                    entity_mask[entity_start_token] = True
                    b.append(entity_mask)

                inputs['entity_mask'] = torch.tensor(b, dtype=torch.bool)
                inputs['priors'] = torch.FloatTensor([ps]).T
                inputs['redirects'] = torch.FloatTensor([res]).T
                inputs['surnames'] = torch.FloatTensor([ss]).T

                input_ids = inputs['input_ids'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                entity_mask = inputs['entity_mask'].to(device)
                adds_redirect = inputs['redirects'].to(device)
                adds_prior = inputs['priors'].to(device)
                adds_surname = inputs['surnames'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, entity_mask=entity_mask, adds_redirect=adds_redirect,
                                adds_surname=adds_surname,
                                token_type_ids=token_type_ids, adds_prior=adds_prior)

                logits = outputs.logits

                preds = logits.detach().cpu().numpy()'''

                f_x = np.exp(preds[preds_idx]) / np.sum(np.exp(preds[preds_idx]))
                prediction = f_x[1]
                if prediction > best_candidate_pred:
                    best_candidate_pred = prediction
                    best_candidate = candidate

                if int(id) == title2id[candidate]:
                    correct_found += 1

                preds_idx += 1

            if int(id) == title2id[best_candidate]:
                correct += 1
                more_candidates['correct'] += 1
            else:
                incorrect += 1
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


def evaluate(loader):
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(loader, desc="Evaluating"):
        model.eval()

        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        entity_mask = batch['entity_mask'].to(device)
        adds_redirect = batch['redirects'].to(device)
        adds_prior = batch['priors'].to(device)
        adds_surname = batch['surnames'].to(device)
        labels = batch['labels'].to(device)



        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, entity_mask=entity_mask, adds_redirect=adds_redirect,
                            adds_surname=adds_surname,
                            token_type_ids=token_type_ids, adds_prior=adds_prior,
                            labels=labels)

            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)


    max_preds = np.argmax(preds, axis=1)
    precision, recall, f1, tp, fp, fn = metrics(max_preds, out_label_ids)
    print("precision: %f, recall: %f, f1: %f" % (precision, recall, f1))
    print("tp: %d, fp: %d, fn: %d" % (tp, fp, fn))

    return preds



#dataset_dev = process(type='dev')
#dataset_test = process(type='test')
#dataset_train = process(type='train')




dataset_dev,test_data_dev = get_dataset(wexea_directory,entity_start_token_id,tokenizer=tokenizer, type='dev')
dataset_test, test_data_test = get_dataset(wexea_directory,entity_start_token_id,tokenizer=tokenizer, type='test')
dataset_train,test_data_train = get_dataset(wexea_directory,entity_start_token_id,tokenizer=tokenizer, type='train')

'''dataset_train = dataset_dev
dataset_test = dataset_dev
test_data_train = test_data_dev
test_data_test = test_data_dev'''

title2id = json.load(open(wexea_directory + 'dictionaries/title2Id.json'))


loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=16, shuffle=False)

weight_decay = 0.0
learning_rate = 1e-5
adam_epsilon = 1e-8

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optim = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

model.to(device)

for epoch in range(EPOCHS):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader_train, leave=True)
    loss_acc = 0.0
    batches = 0.0
    for batch in loop:
        batches += 1.0

        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training

        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        entity_mask = batch['entity_mask'].to(device)
        adds_redirect = batch['redirects'].to(device)
        adds_prior = batch['priors'].to(device)
        adds_surname = batch['surnames'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, entity_mask=entity_mask, adds_redirect=adds_redirect, adds_surname=adds_surname,
                        token_type_ids=token_type_ids, adds_prior=adds_prior,
                        labels=labels)
        # extract loss
        loss = outputs.loss

        loss_acc += loss.item()

        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss_acc / batches)

    preds_dev = evaluate(loader_dev)
    print("test dev")
    run_test(test_data_dev, title2id, preds_dev)

    preds_test = evaluate(loader_dev)
    print("test test")
    run_test(test_data_test, title2id, preds_test)






