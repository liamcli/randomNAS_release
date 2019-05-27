import sys
sys.path.append('/home/lisha/school/Projects/darts_fork/cnn')
import genotypes
from model_search import Network
import utils

import time
import math
import copy
import random
import logging
import os
import gc
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper:
    def __init__(self, save_path, data_dir, seed, batch_size, grad_clip, epochs, init_channels=16, layers=8, learning_rate=0.025, drop_prob=0):
        args = {}
        args['data'] = data_dir
        args['epochs'] = epochs
        args['learning_rate'] = learning_rate
        args['batch_size'] = batch_size
        args['learning_rate_min'] = 0.001
        args['momentum'] = 0.9
        args['weight_decay'] = 3e-4
        args['init_channels'] = init_channels
        args['layers'] = layers
        args['drop_path_prob'] = drop_prob
        args['grad_clip'] = grad_clip
        args['train_portion'] = 0.5
        args['seed'] = seed
        args['log_interval'] = 50
        args['save'] = save_path
        args['gpu'] = 0
        args['cuda'] = True
        args['cutout'] = False
        args['cutout_length'] = 16
        args['report_freq'] = 50
        args = AttrDict(args)
        self.args = args
        self.seed = seed

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.enabled=True
        cudnn.benchmark = False
        cudnn.deterministic=True
        torch.cuda.manual_seed_all(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)


        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        self.train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
          pin_memory=True, num_workers=0)

        self.valid_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
          pin_memory=True, num_workers=0)

        self.train_iter = iter(self.train_queue)
        self.valid_iter = iter(self.valid_queue)

        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        self.criterion = criterion


        try:
            self.load()
            lr = self.scheduler.get_lr()[0]
            print(lr)
            logging.info("Resuming from epoch %d" % self.epochs)
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
        except Exception as e:
            print(e)
            model = Network(args.init_channels, 10, args.layers, self.criterion, drop_prob=args.drop_path_prob)

            model = model.cuda()
            self.model = model

            optimizer = torch.optim.SGD(
              self.model.parameters(),
              args.learning_rate,
              momentum=args.momentum,
              weight_decay=args.weight_decay)
            self.optimizer = optimizer

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
              optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        self.model.zero_grad()

        logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))

        size = 0
        for p in self.model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))

        total_params = sum(x.data.nelement() for x in self.model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

    def train_batch(self, arch):
      args = self.args
      lr = self.scheduler.get_lr()[0]
      self.model.train()

      weights = self.get_weights_from_arch(arch)
      self.set_model_weights(weights)

      step = self.steps % len(self.train_queue)
      input, target = next(self.train_iter)

      n = input.size(0)

      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda()

      if self.steps % len(self.train_queue) == 0:
        self.scheduler.step()
        lr = self.scheduler.get_lr()[0]
        self.objs = utils.AvgrageMeter()
        self.top1 = utils.AvgrageMeter()
        self.top5 = utils.AvgrageMeter()

      #for g in self.optimizer.param_groups:
      #    print(g['weight_decay'], g['momentum'], g['dampening'], g['nesterov'], g['lr'])
      for p in self.model.parameters():
          p.grad = None

      # get a random minibatch from the search queue with replacement
      self.optimizer.zero_grad()
      logits = self.model(input, discrete=True)
      loss = self.criterion(logits, target)
      #print(loss, sum([torch.sum(p.data) for p in self.model.parameters() if p.data is not None]))
      #print(sum([torch.sum(self.optimizer.state[p]['momentum_buffer']) for p in self.optimizer.state if 'momentum_buffer' in self.optimizer.state[p]]))


      loss.backward()
      #print(len([torch.sum(p.grad.data) for p in self.model.parameters() if p.grad is not None]))
      #print(sum([torch.sum(p.grad.data) for p in self.model.parameters() if p.grad is not None]))
      nn.utils.clip_grad_norm(self.model.parameters(), args.grad_clip)
      #print(sum([torch.sum(p.grad.data) for p in self.model.parameters() if p.grad is not None]))

      self.optimizer.step()
      #print(sum([torch.sum(self.optimizer.state[p]['momentum_buffer']) for p in self.optimizer.state if 'momentum_buffer' in self.optimizer.state[p]]))
      #print(sum([torch.sum(p.data) for p in self.model.parameters() if p.data is not None]))
      #print(arch)
      #print(target[:10])
      #print(logits[0])
      #print([p['lr'] for p in self.optimizer.param_groups])
      #print(loss)


      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      self.objs.update(loss.data[0], n)
      self.top1.update(prec1.data[0], n)
      self.top5.update(prec5.data[0], n)

      if step % args.report_freq == 0:
        logging.info('train %03d %e %f %f', step, self.objs.avg, self.top1.avg, self.top5.avg)

      self.steps += 1
      if self.steps % len(self.train_queue) == 0:
        self.epochs += 1
        valid_err = self.evaluate(arch)
        logging.info('epoch %d  |  train_acc %f  |  valid_acc %f' % (self.epochs, self.top1.avg, 1-valid_err))
        self.train_iter = iter(self.train_queue)
        self.model.train()
        self.save()

    def evaluate(self, arch, split=None):
      # Return error since we want to minimize obj val
      logging.info(arch)
      objs = utils.AvgrageMeter()
      top1 = utils.AvgrageMeter()
      top5 = utils.AvgrageMeter()

      self.model.eval()

      weights = self.get_weights_from_arch(arch)
      self.set_model_weights(weights)

      if split is None:
        n_batches = 10
      else:
        n_batches = len(self.valid_queue)

      for step in range(n_batches):
        try:
          input, target = next(self.valid_iter)
        except Exception as e:
          logging.info('looping back over valid set')
          self.valid_iter = iter(self.valid_queue)
          input, target = next(self.valid_iter)
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()

        logits = self.model(input, discrete=True)
        loss = self.criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % self.args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

      return 1-top1.avg

    def save(self):
        checkpoint = {
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'torch_cuda_random_state': torch.cuda.get_rng_state_all(),
                'steps': self.steps,
                'epochs': self.epochs,
                'optimizer': self.optimizer.state_dict(),
                'model': self.model,
                'criterion': self.criterion
                }
        path = os.path.join(self.args.save, 'model.ckpt')
        torch.save(checkpoint, path)

    def load(self):
        args = self.args
        path = os.path.join(args.save, 'model.ckpt')
        checkpoint = torch.load(path)
        #model = Network(args.init_channels, 10, args.layers, self.criterion, drop_prob=args.drop_path_prob)
        #model.load_state_dict(checkpoint['model'])
        model = checkpoint['model']
        self.model = model

        optimizer = torch.optim.SGD(
          self.model.parameters(),
          args.learning_rate,
          momentum=args.momentum,
          weight_decay=args.weight_decay)
        self.optimizer = optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        torch.cuda.set_rng_state_all(checkpoint['torch_cuda_random_state'])
        self.steps = checkpoint['steps']
        self.epochs = checkpoint['epochs']
        print('Resumed model trained for %d steps' % self.steps)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=self.epochs-1)

    def get_weights_from_arch(self, arch):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
        alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

        offset = 0
        for i in range(n_nodes):
            normal1 = arch[0][2*i]
            normal2 = arch[0][2*i+1]
            reduce1 = arch[1][2*i]
            reduce2 = arch[1][2*i+1]
            alphas_normal[offset+normal1[0], normal1[1]] = 1
            alphas_normal[offset+normal2[0], normal2[1]] = 1
            alphas_reduce[offset+reduce1[0], reduce1[1]] = 1
            alphas_reduce[offset+reduce2[0], reduce2[1]] = 1
            offset += (i+2)

        arch_parameters = [
          alphas_normal,
          alphas_reduce,
        ]
        return arch_parameters

    def set_model_weights(self, weights):
      self.model.alphas_normal = weights[0]
      self.model.alphas_reduce = weights[1]
      self.model._arch_parameters = [self.model.alphas_normal, self.model.alphas_reduce]

    def sample_arch(self):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        normal = []
        reduction = []
        for i in range(n_nodes):
            ops = np.random.choice(range(num_ops), 4)
            nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
            nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])

        return (normal, reduction)


    def perturb_arch(self, arch):
        new_arch = copy.deepcopy(arch)
        num_ops = len(genotypes.PRIMITIVES)

        cell_ind = np.random.choice(2)
        step_ind = np.random.choice(self.model._steps)
        nodes_in = np.random.choice(step_ind+2, 2, replace=False)
        ops = np.random.choice(range(num_ops), 2)

        new_arch[cell_ind][2*step_ind] = (nodes_in[0], ops[0])
        new_arch[cell_ind][2*step_ind+1] = (nodes_in[1], ops[1])
        return new_arch


