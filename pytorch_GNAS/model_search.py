import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import logging
import utils
import torchvision.datasets as dset
import numpy as np
import time
import os
class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops) if w.item()>0)


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)


  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._num_ops = len(PRIMITIVES)

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = self.alphas_reduce
      else:
        weights = self.alphas_normal
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=False)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=False)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters



  def genotype_old(self,alphas_normal,alphas_reduce):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) ))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(alphas_normal.data.cpu().numpy())
    gene_reduce = _parse(alphas_reduce.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def genotype(self,alphas_normal,alphas_reduce):

    def _parse(weights):
      gene = []

      index = 0
      for node in range(self._steps):
        count = 0
        for prev_node in range(node+2):
          for op_id in range(self._num_ops):
            if weights[index][op_id] == 1:
              gene.append( (PRIMITIVES[op_id], prev_node ) )
              count += 1
              assert count <= 2


          index += 1

      return gene

    gene_normal = _parse(alphas_normal.data.cpu().numpy())
    gene_reduce = _parse(alphas_reduce.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
class MODEL:
  def __init__(self,
               name,
               args,
               ):
    self.name = name
    self.args = args
    self.criterion = nn.CrossEntropyLoss()
    self.criterion = self.criterion.cuda()
    CIFAR_CLASSES = 10
    self.model = Network(args.init_channels, CIFAR_CLASSES, args.layers, self.criterion)
    self.model = self.model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))

    self.optimizer = torch.optim.SGD(
      self.model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    self.num_train = num_train
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    self.train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.train_batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

    self.valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.valid_batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      self.optimizer, float(args.epochs), eta_min=args.learning_rate_min)


  def train(self,alphas_normal_set,alphas_reduce_set,epoch=0,start_time=0.0):


    num_arcs = len(alphas_normal_set)

    start_time = time.time()
    total_train_steps = len(self.train_queue)
    for step, (input, target) in enumerate(self.train_queue):
      #assign an arc for each mini-batch
      selected_arc = np.random.randint(0,num_arcs)
      self.model.alphas_normal = torch.nn.Parameter(alphas_normal_set[selected_arc].cuda(),
                                               requires_grad=False)
      self.model.alphas_reduce = torch.nn.Parameter(alphas_reduce_set[selected_arc].cuda(),
                                                    requires_grad=False)
      self.model.train()
      n = input.size(0)

      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda(async=True)



      self.optimizer.zero_grad()
      logits = self.model(input)
      loss = self.criterion(logits, target)

      loss.backward()
      nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_clip)
      self.optimizer.step()

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))


      if step % self.args.report_freq == 0:
        lr = self.scheduler.get_lr()[0]
        curr_time = time.time()
        log_string = ""
        log_string += "epoch={:<6d}".format(epoch)
        log_string += "ch_step={:<6d}".format(step+total_train_steps*epoch)
        log_string += " loss={:<8.6f}".format(loss.item())
        log_string += " lr={:<8.4f}".format(lr)
        log_string += " |g|={:<8.4f}".format(utils.compute_grad_norm(self.model))
        log_string += " top_1 tr_acc={:<3d}/{:>3d}".format(int(prec1.item()), n)
        log_string += " top_5 tr_acc={:<3d}/{:>3d}".format(int(prec5.item()), n)
        log_string += " mins={:<10.2f}".format(float(curr_time - start_time) / 60)
        logging.info(log_string)

    utils.save(self.model, os.path.join(self.args.model_path, self.name+'_WS_model.pt'))

    self.scheduler.step()



  def infer(self,alphas_normal_set,alphas_reduce_set):

    self.model.eval()

    num_arcs = len(alphas_normal_set)
    total_valid_steps = len(self.valid_queue)


    valid_acc_list = []

    for arc_id in range(num_arcs):

      self.model.alphas_normal = torch.nn.Parameter(alphas_normal_set[arc_id].cuda(),
                                                    requires_grad=False)
      self.model.alphas_reduce = torch.nn.Parameter(alphas_reduce_set[arc_id].cuda(),
                                                    requires_grad=False)

      input, target = next(iter(self.valid_queue))
      input = Variable(input).cuda()
      target = Variable(target).cuda(async=True)
      with torch.no_grad():
        logits = self.model(input)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      valid_acc = 10.0 * prec1.item() / n
      valid_acc_list.append(valid_acc)


    return valid_acc_list



