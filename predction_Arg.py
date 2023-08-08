# PROGRAMMER:Ezzeddine Almansoob
# DATE CREATED:26/7/2023
# PURPOSE: here we can get all the argemnts from user command line interface

import argparse


def get_ar_p():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to input image')
    parser.add_argument('checkpoint', help='Enter checkpoint.pth with its path')
    parser.add_argument('--top_k', type=int,default=1, help='return top K most likely classes')
    parser.add_argument('--category_names',help='mapping categories from jsonfile ')
    parser.add_argument('--gpu', action='store_true', help='use GPU for inference')

    return parser.parse_args()


