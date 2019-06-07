import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Rank-aware Attention Network")
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)
parser.add_argument('root_path', type=str)

# ============================= Model Configs ================================
parser.add_argument('--num_samples', type=int, default=400)

# =========================== Learning Configs ===============================
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--transform', action='store_true', default=False)
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')

# ============================ Monitor Configs ===============================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--eval-freq', '-ef', default=10, type=int,
                    metavar='N', help='evaluation frequency (default: 10)')

# ============================ Runtime Configs ===============================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="checkpoints/")
parser.add_argument('--run_folder', type=str, default="runs/")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

