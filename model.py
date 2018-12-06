import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(
        torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, device, vocab_size, tag2ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2ix = tag2ix
        self.tagset_size = len(tag2ix)

        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True,
                            batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size).to(device))
        self.transitions.data[self.tag2ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag2ix[STOP_TAG]] = -10000

    def init_hidden(self, batch_sz):
        return (torch.randn(2, batch_sz, self.hidden_dim // 2).to(self.device),
                torch.randn(2, batch_sz, self.hidden_dim // 2).to(self.device))

    def _get_lstm_feature(self, sentence, length):
        length, idx_sort = torch.sort(length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        sentence = sentence[idx_sort]

        batch_sz = sentence.size()[0]
        self.hidden = self.init_hidden(batch_sz)
        embeds = self.word_embed(sentence)
        # batch_sz x max_len x embed_dim
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, length, batch_first=True)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        unpack, length = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # unsort the seq
        # batch_sz x max_len x hidden_dim
        lstm_out = unpack[idx_unsort]
        # batch_sz x max_len x tag_dim
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars =torch.full((1, self.tagset_size), -10000.).to(self.device)
        init_vvars[0][self.tag2ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag2ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag2ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag2ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag2ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag2ix[START_TAG]], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[self.tag2ix[STOP_TAG], tags[-1]]
        return score

    def neg_log_likelihood(self, sentences, pad_tags, lengths):
        batch_feats = self._get_lstm_feature(sentences, lengths)
        batch_loss = torch.zeros(1).to(self.device)
        for i, real_len in enumerate(lengths):
            feats = batch_feats[i][: real_len]
            tags = pad_tags[i][: real_len]
            # print(real_len, feats.size(), tags)
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            batch_loss += forward_score - gold_score
        return batch_loss / lengths.size()[0]

    def forward(self, sentences, lengths):
        batch_feats = self._get_lstm_feature(sentences, lengths)
        scores = []
        tags = []
        for i, length in enumerate(lengths):
            lstm_feats = batch_feats[i][: length]
            score, tag_seq = self._viterbi_decode(lstm_feats)
            scores.append(score)
            tags.append(tag_seq)
        return scores, tags
