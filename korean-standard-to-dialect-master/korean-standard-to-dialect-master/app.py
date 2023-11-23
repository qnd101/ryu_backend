from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import time
import os
import argparse

from data_loader import *
from seq2seq_attn import *
from inference import *
from helper import *
from train import *

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIALECT = 'ryu'
MAX_LENGTH = 110
INPUT_LEVEL = 'syl'
PATH_TRAIN = '../../dataset/dataset/ryutrain.json'
PATH_TEST = '../../dataset/dataset/ryutrain.json'

train_loader = Loader(MAX_LENGTH, INPUT_LEVEL)
test_loader = Loader(MAX_LENGTH, INPUT_LEVEL)
train_loader.readJson(PATH_TRAIN)
test_loader.readJson(PATH_TEST)

SRC = Vocab(train_loader.srcs, INPUT_LEVEL, device)
TRG = Vocab(train_loader.trgs, INPUT_LEVEL, device)
SRC.build_vocab()
TRG.build_vocab()

# from collections import defaultdict
# dict1 = defaultdict(int)
# dict2 = defaultdict(int)
# for src, trg in zip(train_loader.srcs, train_loader.trgs):
#     src_spl = src.split()
#     trg_spl = trg.split()
#     assert len(src_spl) == len(trg_spl), 'Length is different!'
#     dict1[len(src_spl)] += 1
#     dict2[len(trg_spl)] += 1
# for i in range(410):
#     print(i, dict1[i], dict2[i])

train_iterator = train_loader.makeIterator(SRC, TRG, sos=True, eos=True)
test_iterator = test_loader.makeIterator(SRC, TRG, sos=True, eos=True)

portion = int(len(test_iterator) * 0.5)
valid_iterator = test_iterator[:portion]
test_iterator = test_iterator[portion:]

INPUT_DIM = SRC.vocab_size
OUTPUT_DIM = TRG.vocab_size
ENC_EMB_DIM = 128 #256
DEC_EMB_DIM = 128 #256
ENC_HID_DIM = 128 #512
DEC_HID_DIM = 128 #512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 12
CLIP = 1

PAD_IDX = TRG.stoi['<pad>']
SOS_IDX = TRG.stoi['<sos>']
EOS_IDX = TRG.stoi['<eos>']

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SOS_IDX, device, MAX_LENGTH).to(device)                
## model = nn.DataParallel(model)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

model_name = f's2sAttn_{INPUT_LEVEL}_{DIALECT}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}'
model_pt_path = f'../../models/{model_name}/{model_name}.pt'

#print(f'Using cuda : {torch.cuda.get_device_name(0)}')
print(f'Dialect : {DIALECT}')
print(f'Max Length : {MAX_LENGTH}')
print(f'# of train data : {len(train_iterator)}')
print(f'# of test data : {len(test_iterator)}')
print(f'# of valid data : {len(valid_iterator)}')
print(f'SRC Vocab size : {SRC.vocab_size}')
print(f'TRG Vocab size : {TRG.vocab_size}')
print('-' * 20)
print(f'Encoder embedding Dimension : {ENC_EMB_DIM}')
print(f'Decoder embedding Dimension : {DEC_EMB_DIM}')
print(f'Encoder Hidden Dimension : {ENC_HID_DIM}')
print(f'Decoder Hidden Dimension : {DEC_HID_DIM}')
print(f'Encoder dropout rate : {ENC_DROPOUT}')
print(f'Decoder dropout rate : {DEC_DROPOUT}')
print(f'# of epochs : {N_EPOCHS}')
print('-' * 20)
print(f'The model has {count_parameters(model):,} trainable parameters')

try:
    if not os.path.exists(f'../../models/{model_name}'):
        os.makedirs(f'../../models/{model_name}')
except OSError:
    print(f'Failed to create directory : ../../models/{model_name}')

model.load_state_dict(torch.load(model_pt_path))

test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')    

#PATH_LOG = f'./log/sent_{INPUT_LEVEL}_{DIALECT}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}.json'

#test_pair = [[src, trg] for src, trg in zip(test_loader.srcs, test_loader.trgs)]
#result_dict = save_log(PATH_LOG, model, SRC, TRG, test_pair[portion:], INPUT_LEVEL, device)
#result_tuple = [[d['standard'], d['dialect'], d['inference']]for d in result_dict]

#for i in range(1, 5):
#    print(f'{i} {bleu_score(result_tuple, i):.3f}')

print('model is ready!')

app = Flask(__name__)
CORS(app)

links = ['', '', '']

'''
while 1:
    print('>>하이퍼링크를 바꾸러면 link를 입력하세요')
    inp = input('<<')
    if inp == 'link':
        print('>>하이퍼링크 3개를 스페이스로 분리해서 넣어주세요')
        inp = input('<<')
        links = inp.split()
'''
@app.route('/model4')
def model4():
    try:
        # here we want to get the value of user (i.e. ?user=some-value)
        input = request.args.get('input').replace('%', ' ')
        inf = translate_sentence(model, SRC, TRG, input, INPUT_LEVEL, device)
        return jsonify({'text':inf})
    except:
        return {"error": "오류ㅄ"}, 415

@app.get('/model_description')
def model_description():
    text =''
    with open('../../links.txt', 'r', encoding='UTF-8') as f:
        text = f.read()
        f.close()
    links = text.split('&')
    with open('../../descriptions.txt', 'r', encoding='UTF-8') as f:
        text = f.read()
        f.close()
    descriptions = text.split('&')
    return jsonify({'desc':{i+1 : descriptions[i] for i in range(4)}, 'link': {i+1 : links[i] for i in range(4)}})

@app.post('/model_rate')
def new_rating():
    print(request.get_json())
    if request.is_json:
        data = request.get_json()
        with open("../../ratng.txt", "a+") as f:
            f.write(str(data['text']))
            f.write('\n')
            f.close()
        return {}, 201
    return {"error": "Request must be JSON"}, 415