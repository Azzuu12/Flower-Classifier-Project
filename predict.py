# PROGRAMMER:Ezzeddine Almansoob
# DATE CREATED:23/7/2023
# PURPOSE: here we can get all the argemnts from user command line interface

#import related functions and libraries

from predction_Model import process_image, load_checkpoint
from predction_Arg import get_ar_p
import torch
import json
import numpy as np
from PIL import Image


def main():
    #calling get_ar_p() to get the user inputs
    pr_arg = get_ar_p()
    #loading the saved checkpoint if it is exsist
    model = load_checkpoint(pr_arg.checkpoint)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    img_ten = process_image(pr_arg.input)
    image = img_ten.unsqueeze_(0)
    device=torch.device('cuda' if torch.cuda.is_available()and pr_arg.gpu else 'cpu')

    model.to(device)

    #Prediction Process
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        log_ps = model(image)

        ps = torch.exp(log_ps)
        probs, top_class = ps.topk(pr_arg.top_k, dim=1)

    #Option if the user want to use a specified file for mapping names with the classes
    if pr_arg.category_names:
        file_name = pr_arg.category_names
    else:
        file_name = 'cat_to_name.json'
    if pr_arg.top_k == 1:
        prob_pre =np.around(probs.tolist()[0],decimals=5)
        pre_class_s = top_class.tolist()[0]
        prob_re = prob_pre[0]
        pre_class = pre_class_s[0]
        classes = idx_to_class[pre_class]
        with open(file_name, 'r') as f:
            cat_to_name = json.load(f,strict=False)
            flower_name = cat_to_name[classes]
        print('\n\nProbability : {:}\nFlower Name : {}\n'.format(prob_re, flower_name))

    else:
        if pr_arg.category_names:
            prob_pr =np.around(probs.tolist()[0],decimals=5)
            pre_class = top_class.tolist()[0]
            classes = [idx_to_class[idx] for idx in pre_class]
            with open(file_name, 'r') as f:
                cat_to_name = json.load(f,strict=False)
                flower_na = [cat_to_name[i] for i in classes]
            print('\n\nTop {} Probabilities : {:}\n'.format(pr_arg.top_k, prob_pr),
                  '\rList of Probable Flower Names : {}\n'.format(flower_na))
        #Option If the user want the output to demonstrate the class not the name of the flower
        else:
            prob_pr = np.around(probs.tolist()[0],decimals=5)
            pre_class = top_class.tolist()[0]
            classes = [idx_to_class[idx] for idx in pre_class]
            print('\n\nTop {} Probabilities : {:}\n'.format(pr_arg.top_k, prob_pr),
                  '\rList of Probable Classes : {}\n'.format(classes))


if __name__ == "__main__":
    main()
