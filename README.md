# 1 Folder structure
## 1.1 notebooks folder
Use this folder for jupyter notebooks that are created for a first exploration/prototyping of models, preprocessing methods, feature selection, etc.

## 1.2 experiments folder
This folder contains one folder for each experiment that you run. It is not necessary and not advised to manually add files here. A new subfolder is automatically created when you run train.py. It should store best hyperparameters, metrics, and evaluation results. You can also save figures and pickled  models here. 
Best models found by hyperparameter optimization should preferrably saved using the pickle module (for scikit-learn models) or saving methods implemented directly by the library that you took the model from (pytorch and others). 
**Caution**: If you end up saving larger files such as  figures or pickled models, please add them to .gitignore and do not push them to github. 

## 1.3 data
For storing raw and preprocessed data. Do not mix those two subfolders up and do not modify raw data.
**Caution** Do not add preprocessed data to the github as we will end up having too many large files. 

## 1.4 configs
For saving .yaml files for **planned experiments**. The yaml files are used to pass options and parameters to python scripts such as train.py or evaluate.py. Do not modify config files during or after training.

## 1.5 src 
This is where all the source code goes. File names are self-explanatory. 

## 2 Requirements file
requirements.txt is for specifying necessary modules and their versions. **Important: Please update this file in every commmit where you introduce a new module.**

# 3 Running code
1. Make a .yaml config file in the configs folder specifying all preprocessing, training, hyperparameter tuning and validation options
2. From the top project folder, in your command line run `python3 src/sk_tune.py --config configs/your_config_file.yaml''
4. Check that all files were correctly created and filled in the respective experiment directory.
