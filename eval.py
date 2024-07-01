import torch
import argparse
from clearml import Task
import sys

def main(args):
  task = Task.init(project_name='try', task_name='eval')
  print('begin eval...')
  print(f'python version: {sys.version}')
  t1 = torch.Tensor([1, 2, 3])
  print('tensor1', t1)
  print('aa + bb ===', args.aa + args.bb)
  if (args.cc):
    print(f'we have cc: {args.cc}')
  else:
    print(f'we dont have cc')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="enter two N plus")

  parser.add_argument("--aa", required=True, help="First method name that determines which script and environment to use")
  parser.add_argument("--bb", required=True, help="Second method name that determines which script and environment to use")
  parser.add_argument("--cc", help="Third method name that determines which script and environment to use")
  args = parser.parse_args()
  main(args)