# PROGRAMMER:Ezzeddine Almansoob
# DATE CREATED:20/7/2023                              
# PURPOSE: here we can get all the argemnts from user command line interface

import argparse


def get_ar():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to the dataset')
    parser.add_argument('--save_dir', type=str,
                        help='directory to save the checkpoint')
    parser.add_argument('--arch', type=str,default='vgg19',
                        help='Must be vgg19 , alexnet or densenet121')
    parser.add_argument('--learning_rate', type=float,default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int,default=0,
                        help='Number of hidden units')
    parser.add_argument('--epochs', type=int,default=10, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')

    return parser.parse_args()