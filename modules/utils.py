import numpy as np
import torch as t
import torch.nn as nn
from torch.autograd import Variable


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-8):
        super(LayerNorm, self).__init__()

        self.eps = eps

        self.sigma = nn.Parameter(t.ones(size))
        self.mu = nn.Parameter(t.zeros(size))

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = t.mean(z, keepdim=True, dim=-1)
        sigma = t.std(z, keepdim=True, dim=-1)
        out = (z - mu) / (sigma + self.eps)
        out = out * self.sigma.expand_as(out) + self.mu.expand_as(out)

        return out


class PositionWiseNN(nn.Module):
    def __init__(self, size, inner_size, dropout=0.1):
        super(PositionWiseNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(size, inner_size),
            nn.SELU(),
            nn.Linear(inner_size, size)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(size)

    def forward(self, input):
        residual = input

        _, seq_len, size = input.size()
        input = input.view(-1, size)

        result = self.fc(input)
        result = result.view(-1, seq_len, size)
        result = self.dropout(result)

        return self.layer_norm(result + residual)


class PositionalEmbedding(nn.Module):
    def __init__(self, path, vocab_size, max_len, embedding_size):
        super(PositionalEmbedding, self).__init__()

        self.max_len = max_len
        self.embedding_size = embedding_size

        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.positional_embeddings = nn.Embedding(int(max_len), embedding_size, padding_idx=0)

        self.embeddings.weight = nn.Parameter(t.from_numpy(np.load(path)).float(), requires_grad=False)
        self.position_encoding_init()

    def forward(self, input):
        batch_size, seq_len = input.size()

        positional = Variable(t.LongTensor([i for i in range(1, seq_len + 1)])).repeat(batch_size).view(batch_size, -1)
        if input.is_cuda:
            positional = positional.cuda()

        padding_mask = t.eq(input, 0).data
        positional.data.masked_fill_(padding_mask, 0)

        return self.embeddings(input) + self.positional_embeddings(positional)

    def position_encoding_init(self):
        encoding = np.array([
            [pos / np.power(10000, 2 * i / self.embedding_size) for i in range(self.embedding_size)]
            if pos != 0 else np.zeros(self.embedding_size) for pos in range(self.max_len)])

        encoding[1:, 0::2] = np.sin(encoding[1:, 0::2])
        encoding[1:, 1::2] = np.cos(encoding[1:, 1::2])

        self.positional_embeddings.weight = nn.Parameter(t.from_numpy(encoding).float(), requires_grad=False)


class ScheduledOptim(object):
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        self.n_current_steps += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
