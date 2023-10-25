# Flower-Classifier-Project
## Description
Main goal of this project is to build a model that can be trained and after the model is trained it can be used for make prediction. I provide this model with a dataset
flower images so that It can be trained on this dataset and then used to predict the image that will be provided to it.

## CLI Options

## 1. Train
Train a new network on a data set with `train.py`.<br>

  **Basic usage:**<br> `python train.py data_directory`.<br>
  **Options:**<br>
            -Set directory to save checkpoints:`python train.py data_dir --save_dir save_directory`<br>
            -Choose model architecture: `python train.py data_dir --arch "model"` model can be `vgg16`,`alexnet` or `densenet121`<br>
            -Set hyperparameters:`python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`<br>
            - Use GPU for training:`python train.py data_dir --gpu`<br>

            
## 2. Predict
Predict flower name from an image with `predict.py` along with the probability of that name.<br>

  **Basic usage:** <br> `python predict.py /path/to/image checkpoint`<br>
  **Options:** <br>
            -Return topK most likely classes: `python predict.py input checkpoint --top_k 3`<br>
            -Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`<br>
            -Use GPU for inference: `python predict.py input checkpoint --gpu`<br>
            
## Output
-**For training process:** <br>     the output will be the loss of the training and validation datasets along with the accuracy of each.<br>
-**For prediction process:** <br>     the output will be the probability of the flower/s image/s along with the flower/s name/s or class/es

## Important Note 
**in each file in the project files there are comments each steps, you're recommeded to read each of them carefully 
before you go through the code**

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.



