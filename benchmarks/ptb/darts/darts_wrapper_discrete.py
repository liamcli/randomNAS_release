import sys
sys.path.append('/home/liamli4465/darts/rnn')
from model import RNNModel
import genotypes
import data
from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

import time
import math
import numpy as np
import torch
import copy
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import gc

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper:
    def __init__(self, save_path, seed, batch_size, grad_clip, config='eval'):
        if config == 'search':
            args = {'emsize':300, 'nhid':300, 'nhidlast':300, 'dropoute':0, 'wdecay':5e-7}
        elif config == 'eval':
            args = {'emsize':850, 'nhid':850, 'nhidlast':850, 'dropoute':0.1, 'wdecay':8e-7}
        args['config'] = config

        args['data'] = '/home/liamli4465/darts/data/penn'
        args['lr'] = 20
        args['clip'] = grad_clip
        args['batch_size'] = batch_size
        args['search_batch_size'] = 256*4
        args['small_batch_size'] = batch_size
        args['bptt'] = 35
        args['dropout'] = 0.75
        args['dropouth'] = 0.25
        args['dropoutx'] = 0.75
        args['dropouti'] = 0.2
        args['seed'] = seed
        args['nonmono'] = 5
        args['log_interval'] = 50
        args['save'] = save_path
        args['alpha'] = 0
        args['beta'] = 1e-3
        args['max_seq_length_delta'] = 20
        args['unrolled'] = True
        args['gpu'] = 0
        args['cuda'] = True
        args = AttrDict(args)
        self.args = args
        self.seed = seed

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled=True
        torch.cuda.manual_seed_all(args.seed)

        corpus = data.Corpus(args.data)
        self.corpus = corpus

        eval_batch_size = 10
        test_batch_size = 1

        self.train_data = batchify(corpus.train, args.batch_size, args)
        self.search_data = batchify(corpus.valid, args.search_batch_size, args)
        self.val_data = batchify(corpus.valid, eval_batch_size, args)
        self.test_data = batchify(corpus.test, test_batch_size, args)
        self.batch = 0
        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()


        ntokens = len(corpus.dictionary)
        #if args.continue_train:
        #    model = torch.load(os.path.join(args.save, 'model.pt'))
        try:
            model = torch.load(os.path.join(args.save, 'model.pt'))
            print('Loaded model from checkpoint')
        except Exception as e:
            print(e)
            model = RNNModel(ntokens, args.emsize, args.nhid, args.nhidlast,
                   args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute, genotype=genotypes.DARTS)

        size = 0
        for p in model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))
        logging.info('initial genotype:')
        logging.info(model.rnns[0].genotype)

        total_params = sum(x.data.nelement() for x in model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

        self.model = model.cuda()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        #self.parallel_model = model.cuda()

    def set_model_arch(self, arch):
        for rnn in self.model.rnns:
            rnn.genotype = arch

    def train_batch(self, arch):
        args = self.args
        model = self.model
        self.set_model_arch(arch)

        corpus = self.corpus
        optimizer = self.optimizer
        total_loss = self.total_loss

        # Turn on training mode which enables dropout.
        ntokens = len(corpus.dictionary)
        i = self.steps % (self.train_data.size(0) - 1 - 1)
        batch = self.batch

        if i == 0:
            hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
        else:
            hidden = self.hidden

        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

        data, targets = get_batch(self.train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

            # assuming small_batch_size = batch_size so we don't accumulate gradients
            optimizer.zero_grad()
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha > 0:
              loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()
        self.hidden = hidden

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        self.epochs = self.steps // (self.train_data.size(0) - 1 - 1)
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - self.start_time
            val_ppl = self.evaluate(arch)
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | val_ppl {:8.2f}'.format(
                self.epochs, batch % (len(self.train_data) // args.bptt), len(self.train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), val_ppl))
            total_loss = 0
            self.start_time = time.time()
        self.batch += 1
        self.steps += seq_len

    def evaluate(self, arch, n_batches=10):
        # Turn on evaluation mode which disables dropout.
        model = self.model
        #weights = self.get_weights_from_arch(arch)
        self.set_model_arch(arch)
        #self.set_model_weights(weights)
        model.eval()
        args = self.args
        total_loss = 0
        ntokens = len(self.corpus.dictionary)
        hidden = model.init_hidden(self.args.search_batch_size)
        #TODO: change this to take seed so that same minibatch can be used when desired
        #batches = np.random.choice(np.arange(self.search_data.size(0) -1), n_batches, replace=False)
        for i in range(0, self.search_data.size(0) - 1, args.bptt):
            data, targets = get_batch(self.search_data, i, args, evaluation=True)
            targets = targets.view(-1)

            log_prob, hidden = model(data, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

            total_loss += loss * len(data)

            hidden = repackage_hidden(hidden)
        try:
            ppl = math.exp(total_loss[0] / len(self.search_data))
        except Exception as e:
            ppl = 100000
        return ppl
        # TODO: add resume functionality

    def save(self):
        save_checkpoint(self.model, self.optimizer, self.epochs, self.args.save)

    def sample_arch(self):
        n_nodes = genotypes.STEPS
        n_ops = len(genotypes.PRIMITIVES)
        arch = []
        for i in range(n_nodes):
            op = np.random.choice(range(1,n_ops))
            node_in = np.random.choice(range(i+1))
            arch.append((genotypes.PRIMITIVES[op], node_in))
        #concat = [i for i in range(genotypes.STEPS) if i not in [j[1] for j in arch]]
        concat = range(1,9)
        genotype = genotypes.Genotype(recurrent=arch, concat=concat)
        return genotype


    def perturb_arch(self, arch):
        new_arch = copy.deepcopy(arch)
        p = np.arange(1,genotypes.STEPS+1)
        p = p / sum(p)
        c_ind = np.random.choice(genotypes.STEPS, p=p)
        #c_ind = np.random.choice(genotypes.STEPS)
        new_op = np.random.choice(range(1,len(genotypes.PRIMITIVES)))
        new_in = np.random.choice(range(c_ind+1))
        new_arch.recurrent[c_ind] = (genotypes.PRIMITIVES[new_op], new_in)
        #print(arch)
        #arch.recurrent[c_ind] = (arch.recurrent[c_ind][0],new_in)
        return new_arch


    def get_weights_from_arch(self, arch):
        n_nodes = genotypes.STEPS
        n_ops = len(genotypes.PRIMITIVES)
        weights = torch.zeros(sum([i+1 for i in range(n_nodes)]), n_ops)

        offset = 0
        for i in range(n_nodes):
            op = arch[i][0]
            node_in = arch[i][1]
            ind = offset + node_in
            weights[ind, op] = 5
            offset += (i+1)

        weights = torch.autograd.Variable(weights.cuda(), requires_grad=False)

        return weights

