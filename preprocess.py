import pickle as pk
import os
from torch.utils.data import Dataset

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_IDX = 0
UNK_IDX = 1


class DataManager:
    def __init__(self):
        self.tag2ix = {START_TAG: 0, STOP_TAG: 1, 'B-PER': 2, 'I-PER': 3,
                       'B-LOC': 4, 'I-LOC': 5, 'B-ORG': 6, 'I-ORG': 7, 'O': 8}
        self.word2ix = {}
        self.train_sents = []
        self.train_tags = []
        self.dev_sents = []
        self.dev_tags = []
        self.test_sents = []
        self.prepare()

    @staticmethod
    def readfile(name):
        sentences = []
        tags = []
        with open(name, 'r', encoding='utf-8') as f:
            sent, tag = [], []
            for line in f.readlines():
                if not line.strip():
                    if sent:
                        sentences.append(sent)
                        sent = []
                    if tag:
                        tags.append(tag)
                        tag = []
                    continue
                try:  # there are illegal train label like "日 O O, 日 OO"
                    line = line.strip().split('\t')
                    word, t = line[0], line[1]
                    if t == 'OO':
                        t = 'O'
                    sent.append(word)
                    tag.append(t)
                except ValueError:
                    print(line, sum([len(s) for s in sentences]))
        return sentences, tags

    def build_vocab(self, train_sents):
        word2ix = {'PAD': PAD_IDX, 'UNK': UNK_IDX}
        for sent in train_sents:
            for word in sent:
                if word not in word2ix:
                    word2ix[word] = len(word2ix)
        self.word2ix = word2ix
        with open('./word2ix.pk', 'wb') as f:
            pk.dump(self.word2ix, f)

    def prepare(self):
        train_sents, train_tags = self.readfile('./data/train.txt')
        if os.path.exists('./word2ix.pk'):
            with open('word2ix.pk', 'rb') as f:
                self.word2ix = pk.load(f)
            print('load from word2ix.pk')
        else:
            self.build_vocab(train_sents)
        print('vocabulary size: ', len(self.word2ix))

        word2ix = self.word2ix
        self.train_sents = [[word2ix[word] for word in sent]
                            for sent in train_sents]
        self.train_tags = [[self.tag2ix[tag] for tag in tag_list]
                           for tag_list in train_tags]
        print('train set size: ', len(train_sents))

        dev_sents, dev_tags = self.readfile('./data/dev.txt')
        self.dev_sents = [[word2ix[word] if word in word2ix else 0 for word in sent]
                          for sent in dev_sents]
        self.dev_tags = [[self.tag2ix[tag] for tag in tag_list]
                         for tag_list in dev_tags]

        with open('./data/test.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sent = [word2ix[word] if word in word2ix else 0
                        for word in line.strip()]  # UNK is 0
                self.test_sents.append(sent)
        print(line, sent)


class NerDataset(Dataset):
    def __init__(self, sents, tags):
        self.sentences = sents
        self.tags = tags
        self.size = len(sents)
        self.max_len = max([len(s) for s in self.sentences])

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return (self.pad_seq(self.sentences[item]),
                self.pad_seq(self.tags[item]),
                len(self.tags[item]))

    def pad_seq(self, seq):
        padded = [PAD_IDX] * self.max_len
        s_len = len(seq)
        assert s_len <= self.max_len
        padded[: s_len] = seq
        return padded


if __name__ == '__main__':
    data_manager = DataManager()
    trainset = NerDataset(data_manager.train_sents, data_manager.train_tags)
    print(len(trainset), trainset.max_len, trainset[1])
