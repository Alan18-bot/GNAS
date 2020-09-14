import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import *


from game import *
from genotypes import PRIMITIVES


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--train_batch_size', type=int, default=144, help='batch size')
parser.add_argument('--valid_batch_size', type=int, default=100, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--num_arc_class', type=int, default=5, help='num of classes')
parser.add_argument('--num_iterations', type=int, default=150, help='num of training player iterations')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--GNAS_arc_dir', type=str, default='GNAS_Arcs', help='path to save the arcs')
parser.add_argument('--RS_plus_arc_dir', type=str, default='RS_plus_Arcs', help='path to save the arcs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--improved_random_search', action='store_true', default=False, help='perform improved random search')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--player_learning_rate', type=float, default=0.01, help='learning rate for arch encoding')
parser.add_argument('--player_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--player_training_step_per_iter', type=int, default=50, help='num of training player epochs per iteration')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

args.model_path = '{0}/{1}'.format(args.save,args.model_path)
args.GNAS_arc_dir = '{0}/{1}'.format(args.save,args.GNAS_arc_dir)
args.RS_plus_arc_dir = '{0}/{1}'.format(args.save,args.RS_plus_arc_dir)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))



log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)




CIFAR_CLASSES = 10

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  if not os.path.isdir(args.model_path):
    logging.info("Path {} does not exist. Creating.".format(args.model_path))
    os.makedirs(args.model_path)

  if not os.path.isdir(args.GNAS_arc_dir):
    logging.info("Path {} does not exist. Creating.".format(args.GNAS_arc_dir))
    os.makedirs(args.GNAS_arc_dir)
  if not os.path.isdir(args.RS_plus_arc_dir):
    logging.info("Path {} does not exist. Creating.".format(args.RS_plus_arc_dir))
    os.makedirs(args.RS_plus_arc_dir)


  args.seed = np.random.randint(0,100)
  logging.info("----------seed = {}-----------".format(args.seed))
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  GNAS_ws_model = MODEL('GNAS',args)
  RS_plus_ws_model = MODEL('RS_plus',args)
  logging.info("---------------Starting GNAS!---------------")
  GA = Game(GNAS_ws_model,RS_plus_ws_model,args)
  player_prefers = GA.train_search()
  if args.improved_random_search == True:

    logging.info('-------------Starting RS_plus!--------------')
    GA.RS_plus(player_prefers)







if __name__ == '__main__':
  main() 

