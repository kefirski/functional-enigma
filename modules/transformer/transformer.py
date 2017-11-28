import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from utils.beam import Beam
from .decoder import Decoder
from .encoder import Encoder


class Transormer(nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, h_size, k_size, v_size, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(Transormer, self).__init__()

        self.vocab_size = vocab_size

        self.encoder = Encoder(n_layers, n_heads, h_size, k_size, v_size, dropout)
        self.decoder = Decoder(n_layers, n_heads, h_size, k_size, v_size, dropout)

        self.out_fc = nn.Sequential(
            weight_norm(nn.Linear(h_size, 4 * h_size)),
            nn.SELU(),
            weight_norm(nn.Linear(4 * h_size, vocab_size))
        )

    def forward(self, input, decoder_input):
        """
        :param input: An float tensor with shape of [batch_size, encoder_len, h_size]
        :param decoder_input: An float tensor with shape of [batch_size, decoder_len, h_size]
        :return: An float tensor with shape of [batch_size, decoder_len, vocab_size]
        """

        condition = self.encoder(input)
        out = self.decoder(decoder_input, condition)

        batch_size, seq_len, _ = out.size()
        out = out.view(batch_size * seq_len, -1)
        out = self.out_fc(out).view(batch_size, seq_len, -1)

        return out

    def loss(self, input, decoder_input, target, criterion, embeddings, cuda, eval=False):

        if eval:
            self.eval()
        else:
            self.train()

        input = embeddings(input)
        decoder_input = embeddings(decoder_input)

        if cuda:
            input, decoder_input, target = input.cuda(), decoder_input.cuda(), target.cuda()

        out = self(input, decoder_input)
        out = out.view(-1, self.vocab_size)
        target = target.view(-1)

        nll = criterion(out, target) / input.size(0)

        return nll

    def generate(self, input, loader, embeddings, cuda, max_len=40, n_beams=10):

        self.eval()

        input = embeddings(input)
        if cuda:
            input = input.cuda()

        condition = self.encoder(input)

        decoder_input = loader.go_input(1)
        decoder_embed = embeddings(decoder_input)
        if cuda:
            decoder_embed = decoder_embed.cuda()

        beams = [Beam() for _ in range(n_beams)]

        '''
        Starting point for beam search.
        Generate n_beams characters
        '''
        decoder_out = self.decoder(decoder_embed, condition).squeeze(1)
        decoder_out = F.softmax(self.out_fc(decoder_out).squeeze(0), dim=0).data.cpu().numpy()
        samplings = loader.sample_char(decoder_out, n_beams)

        for i, (word, prob) in enumerate(samplings):
            beams[i].update(beams[i].prob * prob, word)

        condition = condition.repeat(n_beams, 1, 1)

        decoder_input = loader.to_tensor([beam.data for beam in beams])
        decoder_embed = embeddings(decoder_input)
        if cuda:
            decoder_embed = decoder_embed.cuda()

        for _ in range(max_len - 1):

            decoder_out = self.decoder(decoder_embed, condition)
            decoder_out = decoder_out[:, -1]
            decoder_out = F.softmax(self.out_fc(decoder_out), dim=1).data.cpu().numpy()

            beams = loader.beam_update(beams, decoder_out)

            decoder_input = loader.to_tensor([beam.data for beam in beams])
            decoder_embed = embeddings(decoder_input)
            if cuda:
                decoder_embed = decoder_embed.cuda()

        return '\n'.join([beam.data for beam in beams])

    def learnable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p
