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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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


def predict(model, sentences, lengths):
    with torch.no_grad():
        scores, pred_tags = model(to_tensor(sentences), to_tensor(lengths))
        tag_label = [ix2tag[t] for row in pred_tags for t in row]
        return tag_label


def eval(model, dev_loader):
    y_true = []
    y_pred = []
    for sents, tags, lengths in tqdm(dev_loader):
        pred_tag = predict(model, sents, lengths)
        true_tag = [ix2tag[t] for row in tags.numpy() for t in row]
        y_true += true_tag
        y_pred += pred_tag
    print(classification_report(y_true, y_pred, 4))


def _main():
    data_manager = DataManager()
    vocab_size = len(data_manager.word2ix)
    model = BiLSTM_CRF(device, vocab_size, data_manager.tag2ix, EMBEDDING_DIM, HIDDEN_DIM)
    model = model.to(device)

    train_set = NerDataset(data_manager.train_sents, data_manager.train_tags)
    dev_set = NerDataset(data_manager.dev_sents, data_manager.dev_tags)
    train_loader = DataLoader(train_set, batch_size=BATCH_SZ, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SZ, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epoch_loss = []

    '''with torch.no_grad():
        precheck_sent = to_tensor(train_loader[0])
        precheck_tag = to_tensor(dataset.train_tags[0])
        print(precheck_tag)
        print(model(precheck_sent))'''

    for epoch in range(EPOCH_NUM):
        for sents, tags, lengths in tqdm(train_loader):
            sents = sents.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)
            # print(lengths, sents.size(), tags.size())
            loss = model.neg_log_likelihood(sents, tags, lengths)

            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, ' epoch loss: ', sum(epoch_loss)/len(epoch_loss))
        save_model(model, epoch)
        eval(model, dev_loader)


if __name__ == '__main__':
    _main()
