import re, json
import torch
import pandas as pd

#Vocabulary 저장 클래스
class Vocab:
    def __init__(self, sentences, input_level, device):
        self.sentences = sentences #클래스에 때려넣을 문장들 
        self.stoi = {'<pad>': 0, '<sos>': 1, '<eos>':2, '<unk>':3} #string to index
        self.s_freq = {'<pad>': 0, '<sos>': 0, '<eos>':0, '<unk>':0} #몇번나왔는지
        self.itos = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'} #index to string
        self.vocab_size = 4 #number of vocabularies
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        self.input_level = input_level
        self.device = device
    
    #단어 하나 추가
    def _addWord(self, s):
        if s not in self.stoi:
            self.stoi[s] = self.vocab_size
            self.s_freq[s] = 1
            self.itos[self.vocab_size] = s
            self.vocab_size += 1
        else:
            self.s_freq[s] += 1

    #모든 문장별로 단어 추가
    def build_vocab(self):
        for sentence in self.sentences:
            if self.input_level == 'syl':
                sentence = [ch for ch in sentence]
            elif self.input_level == 'word':
                sentence = sentence.split()
            elif self.input_level == 'jaso':
                print("NOT IMPLEMENTED!")
                exit()

            for s in sentence:
                self._addWord(s)
    
    #인덱스의 리스트를 받아서 문장으로 바꿔줌
    def sentenceFromIndex(self, indexs):
        ret_list = []
        for t in indexs:
            if t in self.itos:
                ret_list.append(self.itos[t])
            else:   
                ret_list.append('<unk>')
        return ret_list

    #문장을 받아서 인덱스 리스트로 바꿔줌
    def indexesFromSentence(self, sentence, sos, eos):
        ret_list = []
        if sos:
            ret_list.append(self.SOS_IDX)

        if self.input_level == 'syl':
            sentence = [ch for ch in sentence]
        elif self.input_level == 'word':
            sentence = sentence.split()
        elif self.input_level == 'jaso':
            print("NOT IMPLEMENTED!")
            exit()

        for s in sentence:
            if s in self.stoi:
                ret_list.append(self.stoi[s])
            else:
                ret_list.append(self.UNK_IDX)
        if eos:
            ret_list.append(self.EOS_IDX)
        return ret_list

    #열벡터로 변환
    def tensorFromSentence(self, sentence, sos, eos):
        indexes = self.indexesFromSentence(sentence, sos, eos)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)
        # return = [sent len, batch size]
        # return = [sent len, 1]

class Loader:
    def __init__(self, max_len, input_level):
        self.pairs = [] #[표준어, 사투리] 들의 리스트
        self.srcs = [] #표준어
        self.trgs = [] #사투리
        self.max_len = max_len #문장의 최대길이
        self.input_level = input_level

    #문장 정리
    def _normalize(self, sentence):
        if sentence == None:
            return ''
        sentence = sentence.strip()
        sentence = re.sub(r"[^가-힣ㄱ-ㅎ ]", r"", sentence)
        return sentence
    
    #최대길이보다 작은 문장들만 필터함
    def _filterPairs(self):
        filtered = []
        for pair in self.pairs:
            if self.input_level == 'syl':
                len_p0 = len(pair[0])
                len_p1 = len(pair[1])
            if self.input_level == 'word':
                len_p0 = len(pair[0].split())
                len_p1 = len(pair[1].split())

            if len_p0 < self.max_len and len_p1 < self.max_len:
                filtered.append(pair)
        self.pairs = filtered

    def readJson(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        self.pairs = [[self._normalize(d['standard']), self._normalize(d['dialect'])] for d in list(data.values())]
        self._filterPairs()
        self.srcs = [pair[0] for pair in self.pairs]
        self.trgs = [pair[1] for pair in self.pairs]
    
    #SRC, TRG는 Vocab class
    #모든 문장을 텐서로 바꾼 리스트 반환
    def makeIterator(self, SRC, TRG, sos, eos):
        # If sos, eos is True, add <sos>, <eos> tokens.
        ret_list = []
        for pair in self.pairs:
            src = SRC.tensorFromSentence(pair[0], sos, eos)
            trg = TRG.tensorFromSentence(pair[1], sos, eos)
            ret_list.append([src, trg])
        return ret_list
