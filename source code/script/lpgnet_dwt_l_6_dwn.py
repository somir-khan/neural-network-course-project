# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import os
import random
import shutil
# !pip install tqdm==4.64.1
import tqdm
# !pip install numpy==1.19.5
import numpy as np
# !pip install pandas==1.3.5
import pandas as pd
!pip install tensorflow==1.15.0
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.metrics import *

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import tensorflow as tf


import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

# %%
tf.__version__

# %%


# %%

tf.config.run_functions_eagerly = True
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

# %%
sampling_rate = 50
# path = "./dataset/processed100/lp_residual/"
path = "/kaggle/input/dwtlevel6down50/"
# path = "./dataset/raw/"
modelstore = "/kaggle/working/models/"

# %%
walk_df = pd.read_csv("/kaggle/input/gaitdatawithlpr/kaggle_data/WalksDemographics.csv")
walk_df

# %%
data = {}
for i in set(walk_df["Patient ID"]):
    data[i] = []
    
for index,row in tqdm.tqdm(walk_df.iterrows()):
    patient = row["Patient ID"]
#     features = row[["Gender","Age","Height","Weight","TUAG","Speed","Extra Task"]]
    features = row[["Gender","Age","Height","Weight","TUAG"]]

    walk_name = row["Walk Name"]
    walk_seq = np.loadtxt(path+walk_name,delimiter=",")
    sample = (walk_seq,features,row["Class"])
    data[patient].append(sample)

# %%
walk_seq.shape

# %%
patients = sorted(list(data.keys()))
labels = []
for i in patients:
    labels.append(walk_df[walk_df["Patient ID"]==i].iloc[0]["Class"])
#     print (i,labels[-1])

# %%
# Helper Functions

def window(samples,feature,label,cut_length=100):
    inputs = []
    features = []
    labels = []
    
    for i in range(len(samples)):
        sample = samples[i]
        cut = int(cut_length/2)
        for j in range(int(len(sample)/cut)):
            if (j+2)*cut>=len(sample):
                break
            inputs.append(sample[j*cut:(j+2)*cut,:])
            features.append(feature[i])
            labels.append(label[i])
            
    inputs = np.stack(inputs)
    features = np.array(features)
    labels = np.array(labels)
    
    return inputs, features, labels

def pad(samples):
    lengths = [len(i) for i in samples]
    max_len = max(lengths)
    for i in range(len(samples)):
        pad_len = max_len - lengths[i]
        samples[i] = np.pad(samples[i],((0,pad_len),(0,0)),"wrap")
    return np.stack(samples)
    
def get_from_dict(dictionary, keys):
    output = []
    for i in keys:
        output += dictionary[i]
    return output

def get_best_model(path):
    models = os.listdir(path)
    accuracy = {}
    for i in models:
        info = i.split("-")
        try:
            accuracy[float(info[-1][:-5])][float(info[-2])] = i
        except:
            accuracy[float(info[-1][:-5])] = {}
            accuracy[float(info[-1][:-5])][float(info[-2])] = i
    best_acc = max(accuracy.keys())
    best_loss = min(accuracy[best_acc].keys())
    model_path = accuracy[best_acc][best_loss]
    
    model = tf.keras.models.load_model(path+"/"+model_path)
    return model

class VariancePooling(tf.keras.layers.Layer):
    def __init__(self, ):
        super(VariancePooling, self).__init__()

    def call(self, x):
        return tf.math.reduce_std(x,axis=1)

# %%
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
gold = []
predictions = []

AUC = []
accuracy = []
F1Score = []


fold = 0
try:
    shutil.rmtree(modelstore)
except:
    pass

for trainI,testI in kfold.split(patients,labels):
    fold+=1
    train_patients = [patients[i] for i in trainI]
    test_patients = [patients[i] for i in testI]
    print (test_patients)
    
    # Get Walk level data
    train = get_from_dict(data,train_patients)
    test = get_from_dict(data,test_patients)
    
    trainSeq = [i[0] for i in train]
    trainNum = [i[1].values for i in train]
    trainLabel = [i[2] for i in train]

    testSeq = [i[0] for i in test]
    testNum = [i[1].values for i in test]
    testLabel = [i[2] for i in test]
    
    # Normalize Numerical data
    scaler = StandardScaler()
    trainNum = scaler.fit_transform(trainNum)
    testNum = scaler.transform(testNum)
    
    # Impute Numerical data
    imputer = KNNImputer()
    trainNum = imputer.fit_transform(trainNum)
    testNum = imputer.transform(testNum)
    
    # Get Windowed Version of Input
    trainSeqWindows, trainNumWindows, trainLabelWindows = window(trainSeq,trainNum,trainLabel) 
    testSeqWindows, testNumWindows, testLabelWindows = window(testSeq,testNum,testLabel) 

    # Get Padded Version of Input
    trainSeqPadded = pad(trainSeq)
    testSeqPadded = pad(testSeq)
    
    # Define Model
    input_layer = Input(shape=(None,18))
    x = SpatialDropout1D(0.2)(input_layer)
    x = GaussianNoise(0.01)(x)
    x = SeparableConv1D(32,7,activation="linear",kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool1D(2,2)(x)
    x = SeparableConv1D(32,5,activation="linear",kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool1D(2,2)(x)
    x = SeparableConv1D(64,3,activation="linear",kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("elu",name="embedding")(x)
    avg_pool = GlobalAvgPool1D()(x)
    avg_pool = Dropout(0.25)(avg_pool)
    
    prediction = Dense(1,activation="sigmoid",kernel_regularizer=regularizers.l2(0.001),name="det")(avg_pool)
    model = Model(inputs=input_layer, outputs=prediction)
    
    # Windowed Training
    earlystopper = EarlyStopping(monitor="val_loss",mode="min",patience=25,restore_best_weights=True,verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.25,mode="min",patience=14,verbose=1,min_lr=0.0000001)
    checkpointer = ModelCheckpoint(filepath=modelstore+"/backbone/"+str(fold)+"/weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5",
                                  save_best_only=False,monitor="val_acc",save_weights_only=False,verbose=0)
    try:
        os.makedirs(modelstore+"/backbone/"+str(fold))
    except:
        pass
    optimizer = keras.optimizers.Adam(lr=0.0005, clipvalue=0.5)
    loss = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

#     model.compile(optimizer=keras.optimizers.Adam(lr=0.0005,clipvalue=0.5),
#           loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
#           metrics=['accuracy'])

    model.summary()
    model.fit(trainSeqWindows,trainLabelWindows,batch_size=128,epochs=1000,
              validation_data=[[testSeqPadded,testNum],testLabel],
              callbacks=[earlystopper,reduce_lr,checkpointer],verbose=1)
    

#     Load Best model
#     model = get_best_model(modelstore+"/backbone/"+str(fold))
    model.evaluate(testSeqPadded,testLabel)

    
    
    # Define Full Model
    backbone  = Model(inputs=model.input,outputs=model.get_layer("embedding").output)
    backbone.trainable = False
    input_layer = Input(shape=(None,18))
#     input_num = Input(shape = (5))
#     num = Dropout(0.2)(input_num)
#     num = Dense(14,activation="linear",kernel_regularizer=regularizers.l2(0.001))(num)
#     num = BatchNormalization()(num)
#     num = Activation("elu")(num)
    
    embedding = backbone(input_layer)
#     variance_pool = VariancePooling()(embedding)
    avg_pool = GlobalAvgPool1D()(embedding)

    
#     embedding = Concatenate(axis=-1)([avg_pool,num])
    embedding = Dropout(0.25)(avg_pool)
    prediction = Dense(1,activation="sigmoid")(embedding)
    
    full_model = Model(inputs=input_layer, outputs=prediction)

    # Train Full Model
    earlystopper = EarlyStopping(monitor="val_loss",mode="min",patience=25,restore_best_weights=True,verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.25,mode="min",patience=14,verbose=1,min_lr=0.0000001)
    checkpointer = ModelCheckpoint(filepath=modelstore+"/full/"+str(fold)+"/weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5",
                                  save_best_only=False,monitor="val_acc",save_weights_only=False,verbose=0)
    try:
        os.makedirs(modelstore+"/full/"+str(fold))
    except:
        pass
    
    full_model.compile(optimizer=keras.optimizers.Adam(lr=0.001,clipvalue=1),
          loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
          metrics=['accuracy'])
    full_model.summary()
    full_model.fit(trainSeqPadded,trainLabel,batch_size=64,epochs=200,
          validation_data=[testSeqPadded,testLabel],
          callbacks=[earlystopper,reduce_lr,checkpointer],verbose=1)
    
#     Load Best Model
#     full_model = get_best_model(modelstore+"/full/"+str(fold))

    # Test 
    full_model.evaluate([testSeq],testLabel)
    
    prob = []
    binary = []
    for i in range(len(testSeq)):
        pred = full_model.predict([[testSeq[i]]])[0][0]
        prob.append(pred)
        if pred>0.5:
            binary.append(1)
        else:
            binary.append(0)
            
            
    predictions.append(prob)
    gold.append(testLabel)
        
    AUC.append(roc_auc_score(testLabel,prob))
    accuracy.append(accuracy_score(testLabel,binary))
    F1Score.append(f1_score(testLabel,binary))
    
    print (AUC[-1],accuracy[-1],F1Score[-1])
    # Cleanup
    #     del full_model
    #     del model
    #     break
    
print ("AUC\t\t",np.mean(AUC),np.std(AUC))
print ("Accuracy\t",np.mean(accuracy),np.std(accuracy))
print ("F1Score\t\t",np.mean(F1Score),np.std(F1Score))
print ("All_AUC\t\t",AUC)
print ("All_Accuracy\t",accuracy)
print ("All_F1Score\t\t",F1Score)

# %%


# %%



