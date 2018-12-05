import torch
import torch.optim as optim
from tqdm import tqdm
from os import path, mkdir
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader

from preprocess import *
from model import BiLSTM_CRF

EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCH_NUM = 10
BATCH_SZ = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ix2tag = ['<START>', '<STOP>', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']


def to_tensor(seq):
    return torch.tensor(seq, dtype=torch.long).to(device)


def save_model(model, epoch):
    m_dir = path.join(path.dirname(__file__), 'weights')
    if not path.exists(m_dir):
        mkdir(m_dir)
    torch.save(model.state_dict(), path.join(m_dir, 'epoch%d.h5' % epoch))


def load_model(model, epoch):
    filename = path.join(path.dirname(__file__), 'weights/epoch%d.h5' % epoch)
    if not path.exists(filename):
        print("File not found!", filename)
    else:
        model.load_state_dict(torch.load(filename))


def predict(model, sentence):
    with torch.no_grad():
        score, pred_tag = model(to_tensor(sentence))
        tag_label = [ix2tag[t] for t in pred_tag]
        return tag_label


def eval(model, dev_sents, dev_tags):
    y_true = []
    y_pred = []
    for i, sent in enumerate(dev_sents):
        pred_tag = predict(model, sent)
        true_tag = [ix2tag[t] for t in dev_tags[i]]
        y_true += true_tag
        y_pred += pred_tag
    print(classification_report(y_true, y_pred, 4))


def _main():
    data_manager = DataManager()
    vocab_size = len(data_manager.word2ix)
    model = BiLSTM_CRF(device, vocab_size, dataset.tag2ix, EMBEDDING_DIM, HIDDEN_DIM)
    model = model.to(device)

    trainset = NerDataset(data_manager.train_sents, data_manager.train_tags)
    train_loader = DataLoader(trainset, batch_size=BATCH_SZ, shuffle=True)

    for sents, tags, lengths in train_loader:
        print(lengths, sents, tags)
        print(sents.size(), tags.size())
        break

    with torch.no_grad():
        precheck_sent = to_tensor(train_loader[0])
        precheck_tag = to_tensor(dataset.train_tags[0])
        print(precheck_tag)
        print(model(precheck_sent))

    load_model(model, 0)
    eval(model, dataset.dev_sents, dataset.dev_tags)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(EPOCH_NUM):
        train_sz = len(dataset.train_sents)
        shuffle_index = torch.randperm(train_sz)
        epoch_loss = []
        batch_loss = torch.zeros(1).to(device)
        for i in tqdm(range(train_sz)):
            idx = shuffle_index[i]
            sent_in = to_tensor(dataset.train_sents[idx])
            tag_in = to_tensor(dataset.train_tags[idx])
            loss = model.neg_log_likelihood(sent_in, tag_in)
            batch_loss += loss
            if (i + 1) % BATCH_SZ == 0:
                batch_loss = batch_loss / BATCH_SZ
                epoch_loss.append(batch_loss.item())
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                batch_loss = torch.zeros(1).to(device)
        print('epoch loss: ', sum(epoch_loss)/len(epoch_loss))
        save_model(model, epoch)
        eval(model, dataset.dev_sents, dataset.dev_tags)



if __name__ == '__main__':
    _main()