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

## 1.6 Results
This folder is only for saving final models for submission trained on all data and the other deliverables for the assingment.

## 2 Requirements file
requirements.txt is for specifying necessary modules and their versions. **Important: Please update this file in every commmit where you introduce a new module.**

## 3 Suggested Workflow
1. **Experimentation**: Use the notebooks to explore models, preprocessing or feature selection techniques. Look at the numbers, make plots, use unsupervised methods to cluster, etc. 
2. **Model implementation**: Once you have found a model that you deem worth trying in a more systematic way, implement it in the src/models.py file (or src/feature_selection.py, etc., depending on your goal). If you plan to use just a scikit-learn model, don't implement anything in models.py and just add it to model_registry.py.
3. **Experimental setup**: Create a .yaml file in the configs folder with all the necessary parameters for the experiment. An example of what this should look like is in the config folder.
4. **Preprocessing**: Execute any necessary preprocessing steps and save preprocesed data to data/preprocessed. Do **never** modify raw data.
5. **Training**: Find the best hyperparameters, for example using scikit-learns GridsearchCV(). K-fold cross-validation is strongly recommended. You should at least save the winning set of parameters as pickle file.
6. **Evaluation**: Evaluate the best model that you have found on multiple train-test-splits to obtain an average test error. **Minimum set of metrics** required are the root mean-squared-error and the pearson correlation coefficient, implemented automatically by the function compute_metrics() in src/utils.py. Feel free to add more metrics.
4. **Documentation**: Once you run an experiment, the corresponding .yaml file will automatically be saved in the experiments folder. Make sure to document relevant information about the experiment in the notes.md file contained in the experiments folder.
