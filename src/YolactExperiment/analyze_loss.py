import os
import sys
import argparse
sys.path.append("../")
import TrainingUtils
parser = argparse.ArgumentParser(
    description='Yolact analyze loss Script')
parser.add_argument('--log_folder', default=None, type=str,
                    help='folder with logs, or multiple folder separated with comma')
parser.add_argument('--names', default="1,2,3,4,5,6,7,8,9,10", type=str,
                    help='name for a loss, or multiple names separated with comma')
parser.add_argument('--color', default="blue, red, green", type=str,
                    help='loss line color, or multiple colors separated with comma')
parser.add_argument('--range', default=2, type=int,
                    help='loss will be ploted in the interval (0,--range)')
parser.add_argument('--precision', default=2, type=int,
                    help='loss precision, each --precision iteration will be averaged before plotting')
parser.set_defaults(keep_latest=False)
args = parser.parse_args()
if __name__ == '__main__':
    folders  = args.log_folder.replace(" ","").split(",")
    folders = [ i + os.path.sep + 'Euclidea' for i in folders]
    colors = args.color.replace(" ","").split(",")
    names = args.names.replace(" ","").split(",")
    assert len(colors) >= len(folders) and len(names) >= len(folders)
    TrainingUtils.yolact_training_stats(folders, color=colors, loss_legend=names, y_range=args.range, precision=args.precision)