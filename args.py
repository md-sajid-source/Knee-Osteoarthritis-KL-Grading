import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-backbone', default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('-dataset_dir',type=str,default='OSAIL_KL_Dataset/Labeled')
    parser.add_argument('-csv_dir', default='data/CSVs')
    parser.add_argument('-output_dir', default='sessions')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-bs', type=int, default=16)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-wd', type=float, default=0.00001)
    return parser.parse_args()
