%matplotlib inline
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook
from keras_tqdm import TQDMNotebookCallback
from keras_radam import RAdam
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,classification_report
from multiclass_modelbuild import model_build, model_train, model_save, model_load, plot_roc, plot_confusion_matrix
from ecg_preprocessing import val_split, load_cardiologist_test_set,multiclass_val_split,all_split
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem, t

import os
import json
from collections import Counter
from datetime import date
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import time
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

tf.test.is_gpu_available()

# Load parameters
params = json.load(open('config.json', 'r'))

# Load data and label
all_id, all_label = all_split(21,params)

# build model
model, parallel_model = model_build(params)

# model training (we here train the model for 10 times to calculate the mean and CIs)
for j in tqdm_notebook(range(10)):
    val_id = all_id[j]
    val_label = all_label[j]
    train_id =[]
    train_label = {}
    for i in tqdm_notebook(range(10)):
        if i !=j: 
            train_id  = train_id + all_id[i]
            train_label.update(all_label[i]) 
    model, parallel_model = model_build(params)
    model, parallel_model = model_train(model, parallel_model, train_id, train_label, val_id, val_label, params)
    model.save('multilabel_model_' + str(j+1) + '.h5') #save model
    time.sleep(1800)
    



