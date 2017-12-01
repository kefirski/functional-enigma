import argparse

import torch as t
import torch.nn as nn
from torch.optim import Adam

from modules.transformer.transformer import Transormer
from modules.utils import PositionalEmbedding
from modules.utils import ScheduledOptim
from utils.news_dataloader import Dataloader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='octo-enigma')
    parser.add_argument('--num-iterations', type=int, default=250_000, metavar='NI',
                        help='num iterations (default: 250_000)')
    parser.add_argument('--batch-size', type=int, default=2, metavar='BS',
                        help='batch size (default: 5)')
    parser.add_argument('--steps', type=int, default=15, metavar='S',
                        help='num steps before optimization step (default: 80)')
    parser.add_argument('--num-threads', type=int, default=4, metavar='BS',
                        help='num threads (default: 4)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='D',
                        help='dropout rate (default: 0.1)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    parser.add_argument('--tensorboard', type=str, default='default_tb', metavar='TB',
                        help='Name for tensorboard model')
    args = parser.parse_args()

    t.set_num_threads(args.num_threads)
    loader = Dataloader('~/projects/functional-enigma/utils/data/', '~/projects/wiki.en.bin')

    transformer = Transormer(loader.vocab_size, 8, 14, 300, 45, 45, dropout=args.dropout)
    embed = PositionalEmbedding(loader.preprocessed_embeddings, loader.vocab_size, 125, 300)
    if args.use_cuda:
        transformer = transformer.cuda()

    optimizer = ScheduledOptim(Adam(transformer.learnable_parameters(), betas=(0.9, 0.98)), 300, 10000)

    crit = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    print('Model have initialized')

    for i in range(args.num_iterations):

        out = 0
        for step in range(args.steps):
            input, decoder_input, target = loader.torch(args.batch_size, volatile=False)
            nll = transformer.loss(input, decoder_input, target, crit, embed, args.use_cuda)
            nll /= args.steps
            out += nll.cpu().data
            nll.backward()

        optimizer.step()
        optimizer.update_learning_rate()
        optimizer.zero_grad()

        if i % 25 == 0:
            print('i {}, nll {}'.format(i, out.numpy()))
            print('_________')

        if i % 50 == 0:
            input, _, target = loader.torch(1, volatile=True)
            print(' '.join([loader.idx_to_word[idx] for idx in input[0].cpu().data.numpy()]))
            print('_________')
            print(' '.join([loader.idx_to_word[idx] for idx in target[0].cpu().data.numpy()]))
            print('_________')
            print(transformer.generate(input, loader, embed, args.use_cuda, n_beams=3))
            print('_________')

        if (i + 1) % 5000 == 0:
            t.save(transformer.cpu().state_dict(), args.save)
            transformer = transformer.cuda()
