import numpy as np
import pandas as pd
import xlrd
from tqdm import tqdm_notebook
from multiprocessing import Process, Manager, Pool
import os
import datetime
from sklearn.utils import shuffle
from keras.utils import to_categorical
import random

class load_ecg:
    def __init__(self, ecg_dir):
        manager = Manager()
        self.data = manager.list()
        self.i_list = manager.list()
        self.file_path = ecg_dir
        self.file_list = os.listdir(self.file_path)

    def read_data_with_skiprows(self, work):
        wb = xlrd.open_workbook(work[1] + self.file_list[work[0]], logfile=open(os.devnull, 'w'))
        temp = np.array(pd.read_excel(wb, skiprows=[0, 1, 2], engine='xlrd'))
        self.data.append(temp)
        self.i_list.append(work[0])

    def read_data_without_skiprows(self, work):
        wb = xlrd.open_workbook(work[1] + self.file_list[work[0]], logfile=open(os.devnull, 'w'))
        temp = np.array(pd.read_excel(wb, engine='xlrd'))
        self.data.append(temp)
        self.i_list.append(work[0])

    def load_data(self, saved_name, normalized, skiprows, save):
        def normalize(data):
            mx = np.max(data, axis=0)
            mn = np.min(data, axis=0)
            return (data - mn) / (mx - mn)

        print('Start loading data from ' + self.file_path + '...')
        items = zip([x for x in range(len(self.file_list))], [self.file_path] * len(self.file_list))
        p = Pool(30)
        if skiprows == True:
            list(tqdm_notebook(p.imap(self.read_data_with_skiprows, items), total=len(self.file_list)))
        else:
            list(tqdm_notebook(p.imap(self.read_data_without_skiprows, items), total=len(self.file_list)))
        p.close()
        p.join()
        print('Loading data done...')

        data_set = [x for x in range(len(self.file_list))]
        for i in range(len(self.file_list)):
            data_set[self.i_list[i]] = self.data[i]

        if normalized == True:
            print("Start normalizing...")
            for i in tqdm_notebook(range(np.array(data_set).shape[0])):
                data_set[i] = normalize(data_set[i])
            print("Normalizing done...")

        if save == True:
            print('Start saving files...')
            np.save(saved_name + '_' + datetime.date.today().strftime("%m%d"), data_set)
            np.save(saved_name + '_norm_' + datetime.date.today().strftime("%m%d"), norm_data_set)
            np.save(saved_name + '_filelist', self.file_list)
            print('Saving files done...')

        return np.array(data_set), norm_data_set, self.file_list

def get_dataset(dir_list, saved_name_list, skiprows, saved_bool, normalized=True):    
    for i in tqdm_notebook(range(len(dir_list))):
        if skiprows[i] == True:
            data_loader = load_ecg(dir_list[i])
            data_loader.load_data(saved_name_list[i],  
                                 normalized=normalized, skiprows=True, save=saved_bool[i])
        else:
            data_loader = load_ecg(dir_list[i])
            data_loader.load_data(saved_name_list[i],  
                                 normalized=normalized, skiprows=False, save=saved_bool[i])

def multilabel_ecg_loader(ecg_dir, label_dir, n_classes):
    def random_split(data_set, file_list):
        index = np.arange(len(data_set))
        np.random.shuffle(index)
        n_test = int(len(data_set) * 0.05)
        test_index = index[:n_test]
        train_index = index[n_test:]
        test = data_set[test_index]
        train = data_set[train_index]
        test_file_list = file_list[test_index]
        train_file_list = file_list[train_index]
        return train, train_file_list, test, test_file_list
    
    def over_sampling(data_set, file_list, n_target):
        delta = n_target - data_set.shape[0]
        fold = delta // len(data_set)
        index = np.repeat(np.arange(len(data_set)), fold + 2)
        np.random.shuffle(index)
        np.random.shuffle(index)
        np.random.shuffle(index)
        return data_set[index[:n_target]], file_list[index[:n_target]]
    
    def multilabel_marker(target, file_list_dirs, n_classes):
        note_taker = []
        file_list = np.array([])
        for i in range(n_classes):
            note_taker.append(np.load(file_list_dirs[i]))
            file_list = np.append(file_list, note_taker[i])

        label = np.zeros((target.shape[0], n_classes))
        for i in range(target.shape[0]):
            for j in range(n_classes):
                if target[i] in note_taker[j]:
                    label[i][j] = 1

        return label
    
    def nan_sweeper(data_set, label):
        nan_index = []
        for i in range(data_set.shape[0]):
            if np.isnan(data_set[i]).any():
                nan_index.append(i)
        data_set = np.delete(data_set, nan_index, axis=0)
        label = np.delete(label, nan_index, axis=0)

        return data_set, label
    

    x_train = np.array([])
    x_test = np.array([])
    y_train = np.array([])
    y_test = np.array([])
    
    print('Starting loading ECGs...')
    for i in tqdm_notebook(range(len(ecg_dir))):
        print('Loading ECGs from ' + ecg_dir[i] + '...')
        data_loader = np.load(ecg_dir[i], mmap_mode='r')
        label_dir_loader = np.load(label_dir[i], mmap_mode='r')
        train, train_label_dir, test, test_label_dir = random_split(data_loader, label_dir_loader)
        if train.shape[0] < 10000:
            train, train_label_dir = over_sampling(train, train_label_dir, n_target=10000)
        x_train = np.append(x_train, train)
        x_test = np.append(x_test, test)
        y_train = np.append(y_train, multilabel_marker(train_label_dir, label_dir, n_classes))
        y_test = np.append(y_test, multilabel_marker(test_label_dir, label_dir, n_classes)) 
        
    x_train, y_train = nan_sweeper(x_train.reshape(-1, 5000, 12), y_train.reshape(-1, n_classes))
    x_test, y_test = nan_sweeper(x_test.reshape(-1, 5000, 12), y_test.reshape(-1, n_classes))
    
    x_train, y_train = shuffle(x_train, y_train)
    print('Loading ECGs done...')
    
    return x_train, y_train, x_test, y_test

def multiclasses_ecg_loader(ecg_dir):
    def random_split(data_set):
        index = np.arange(len(data_set))
        np.random.shuffle(index)
        n_test = int(len(data_set) * 0.05)
        test_index = index[:n_test]
        train_index = index[n_test:]
        test = data_set[test_index]
        train = data_set[train_index]
        return train,test
    
    def over_sampling(data_set, n_target):
        delta = n_target - data_set.shape[0]
        fold = delta // len(data_set)
        index = np.repeat(np.arange(len(data_set)), fold + 2)
        np.random.shuffle(index)
        np.random.shuffle(index)
        np.random.shuffle(index)
        return data_set[index[:n_target]]
    

    
    def nan_sweeper(data_set, label):
        nan_index = []
        for i in range(data_set.shape[0]):
            if np.isnan(data_set[i]).any():
                nan_index.append(i)
        data_set = np.delete(data_set, nan_index, axis=0)
        label = np.delete(label, nan_index, axis=0)

        return data_set, label
    

    x_train = np.array([])
    x_test = np.array([])
    y_train = np.array([])
    y_test = np.array([])
    
    print('Starting loading ECGs...')
    for i in tqdm_notebook(range(len(ecg_dir))):
        print('Loading ECGs from ' + ecg_dir[i] + '...')
        data_loader = np.load(ecg_dir[i], mmap_mode='r')
   
        train, test = random_split(data_loader)
        if train.shape[0] < 10000:
            train = over_sampling(train,  n_target=10000)
        x_train = np.append(x_train, train)
        x_test = np.append(x_test, test)
        y_train = np.append(y_train, np.ones(train.shape[0], dtype=np.int64)*i)
        y_test = np.append(y_test, np.ones(test.shape[0], dtype=np.int64) *i)
    y_train = to_categorical(y_train, num_classes=len(ecg_dir))
    y_test = to_categorical(y_test, num_classes=len(ecg_dir))
    
    x_train, y_train = nan_sweeper(x_train.reshape(-1, 5000, 12), y_train.reshape(-1, len(ecg_dir)))
    x_test, y_test = nan_sweeper(x_test.reshape(-1, 5000, 12), y_test.reshape(-1, len(ecg_dir)))
    
    x_train, y_train = shuffle(x_train, y_train)
    print('Loading ECGs done...')
    
    return x_train, y_train, x_test, y_test

def val_choose(num, data):
    xmlfolder = os.listdir('/mnt/data/ECG_data/ECG_xml')
    datafolder = os.listdir('/mnt/data/ECG_data/ECG')
    xmlfolder.sort()
    datafolder.sort()
    xml_list = os.listdir('/mnt/data/ECG_data/ECG_xml/'+xmlfolder[num])
    data_list = os.listdir('/mnt/data/ECG_data/ECG/'+datafolder[num])
    dic = {}
    data_batch=  []
    #ecg from GE machine
    if num ==0:
        random.shuffle(data_list)
        for i in data_list:
            if len(data_batch)<=len(data_list)*0.1:
                data_batch.append(i[:-5]+'.npy')
            else:
                break
        return data_batch
    for l in xml_list:
        s = l[:-4].split('_')
        dic[s[1]] = dic.get(s[1],[])
        dic[s[1]].append(s[0]+'ECG.npy' )
    
    values = list(dic.values())
    random.shuffle(values)
    #xml file corresponding to ECG file from GE machine
    for value in values:
        if len(data_batch)<=len(data_list)*0.1:
            for i in value:
                if i in data:
                    data_batch.append(i)
        else:
            break
    else:
    # ecg from Holder machine   
        dic = {}
        for i in data_list:
            if "_" in i:
                s = i[:-4].split('_')
                dic[s[0]] = dic.get(s[0],[])
                dic[s[0]].append(i[:-4]+'.npy')
            elif "-" in i:
                s = i[:-5].split('-')
                dic[s[0]] = dic.get(s[0],[])
                dic[s[0]].append(i[:-5]+'.npy')
        values = list(dic.values())
        random.shuffle(values)
        for value in values:
            if len(data_batch)<=len(data_list)*0.1:
                for i in value:
                    if i in data:
                        data_batch.append(i)
            else:
                break
        else:
            print(len(data_batch))
            print(datafolder[num])
            print(xmlfolder[num])
            raise 
    return data_batch

def val_split(class_num, params ):
    data = os.listdir(params["multilabel_data_folder"])
    val_data = []
    for i  in tqdm_notebook(range(class_num)):
        val_data = val_data + val_choose(i,data)
    for i in tqdm_notebook(val_data):
        if i in data:
            data.remove(i)
        else:
            val_data.remove(i)
            
    train_label = np.array([np.load(params["multilabel_label_folder"] + i) for i in tqdm_notebook(data)])
    val_label = np.array([np.load(params["multilabel_label_folder"] + i) for i in tqdm_notebook(val_data)])
 
    train_id = np.array([params["multilabel_data_folder"] + i for i in data]).reshape(-1, 1)
    val_id = np.array([params["multilabel_data_folder"] + i for i in val_data]).reshape(-1, 1)
    
    train_id = train_id.reshape(-1).tolist()
    val_id = val_id.reshape(-1).tolist()
    
    train_label = dict(zip(train_id, train_label))
    val_label = dict(zip(val_id, val_label))

    return train_id, train_label, val_id, val_label

def multiclass_val_split(class_num, params ):
    data = os.listdir(params["multilabel_data_folder"])
    val_data = []
    for i  in tqdm_notebook(range(class_num)):
        val_data = val_data + val_choose(i,data)
    for i in tqdm_notebook(val_data):
        if i in data:
            data.remove(i)
        else:
            val_data.remove(i)
            
    train_label =  np.load(params["multilabel_label_folder"] + data[0])
    train_data_onehot = []
    print('loading train_label')
    for i in tqdm_notebook(data):
        temp_label = np.load(params["multilabel_label_folder"] + i)
        
        if sum(temp_label)==1:
            train_label = np.vstack((train_label,temp_label))
            train_data_onehot.append(i)
    val_data_onehot = []        
    val_label = np.load(params["multilabel_label_folder"] + val_data[0])  
    print('loading val_label')
    for i in tqdm_notebook(val_data):
        temp_label = np.load(params["multilabel_label_folder"] + i)
        if sum(temp_label) ==1:
            val_label = np.vstack((train_label,temp_label))
            val_data_onehot.append(i)

    
#     train_label = np.delete(train_label,0,axis=0)
#     val_label = np.delete(val_label,0,axis=0)
    
    train_id = np.array([params["multilabel_data_folder"] + i for i in train_data_onehot]).reshape(-1, 1)
    val_id = np.array([params["multilabel_data_folder"] + i for i in val_data_onehot]).reshape(-1, 1)
    
    train_id = train_id.reshape(-1).tolist()
    val_id = val_id.reshape(-1).tolist()
    
    train_label = dict(zip(train_id, train_label))
    val_label = dict(zip(val_id, val_label))

    return train_id, train_label, val_id, val_label
#    9:1
def all_choose(num, data):
    xmlfolder = os.listdir('/mnt/data/ECG_data/ECG_xml')
    datafolder = os.listdir('/mnt/data/ECG_data/ECG')
    xmlfolder.sort()
    datafolder.sort()
    xml_list = os.listdir('/mnt/data/ECG_data/ECG_xml/'+xmlfolder[num])
    data_list = os.listdir('/mnt/data/ECG_data/ECG/'+datafolder[num])
    dic = {}
    conter = 0
    data_batch = []
    for i in range(10):
        data_batch.append([])
    normdatasum = 0
    tensecondsdatasum = 0
    print(datafolder[num])
    #ecg from GE machine
    if num ==0:
        random.shuffle(data_list)
        for i in data_list:
            while len(data_batch[conter%10])>len(data_list)*0.1:
                conter =conter +1
            else:
                data_batch[conter%10].append(i[:-5]+'.npy')
                normdatasum =normdatasum+1
                conter =conter +1
        
        return data_batch
    
    for l in xml_list:
        s = l[:-4].split('_')
        dic[s[1]] = dic.get(s[1],[])
        dic[s[1]].append(s[0]+'ECG.npy')
    
    values = list(dic.values())
    random.shuffle(values)
    #xml file corresponding to ECG file from GE machine
    for value in values:
        while len(data_batch[conter%10])>len(data_list)*0.1:
            conter =conter +1
        else:
            for i in value:
                if i in data:
                    data_batch[conter%10].append(i)
                    normdatasum =normdatasum+1
            conter =conter +1

    else:
    # ecg from Holter machine  
        dic = {}
        for i in data_list:
            if "_" in i:
                s = i[:-4].split('_')
                dic[s[0]] = dic.get(s[0],[])
                dic[s[0]].append(i[:-4]+'.npy')
            elif "-" in i:
                s = i[:-5].split('-')
                dic[s[0]] = dic.get(s[0],[])
                dic[s[0]].append(i[:-5]+'.npy')
        values = list(dic.values())
        random.shuffle(values)
        for value in values:
            while len(data_batch[conter%10])>len(data_list)*0.1:
                conter =conter +1
            else:
                for i in value:
                    if i in data:
                        data_batch[conter%10].append(i)
                        tensecondsdatasum = tensecondsdatasum+1
                conter =conter +1
#         else:
#             print(len(data_batch))
#             print(datafolder[num])
#             print(dic)
#             raise 

    return data_batch

def all_split(class_num, params ):
    data = os.listdir('/mnt/data/ECG_data/multilabel/data')
    all_data = []
    for i in range(10):
        all_data.append([])
    all_label = []
    for i in range(10):
        all_label.append([])
    all_id = []
    for i in range(10):
        all_id.append([])
    temp = []
    for i  in tqdm_notebook(range(class_num)):
        
        temp = all_choose(i,data)
        for j  in tqdm_notebook(range(10)): 
            
            for onedata in temp[j]:
                if onedata not in  data:
                    temp[j].remove(onedata)
                    print('warning:',onedata)
                
            all_data[j] = all_data[j]+ temp[j]
    
    for i in tqdm_notebook(range(10)):
        
        all_id[i] =  np.array([params["multilabel_data_folder"] + i for i in all_data[i]]).reshape(-1, 1)
        all_id[i] = all_id[i].reshape(-1).tolist()
        all_label[i] = np.array([np.load(params["multilabel_label_folder"] + i) for i in all_data[i]])
        all_label[i] = dict(zip(all_id[i], all_label[i]))
        
    return all_id, all_label

def load_cardiologist_test_set(data_dir, label_dir):
    ctd = [data_dir + i for i in sorted(os.listdir(data_dir))]
    ctl = [np.load(label_dir + i) for i in sorted(os.listdir(label_dir))]
    ctl = dict(zip(ctd, ctl))
    
    X_cdolg_test = []
    y_cdolg_test = []
    for i in ctd:
        X_cdolg_test.append(np.load(i))
    for i in range(len(ctd)):
        y_cdolg_test.append(ctl[ctd[i]])
    
    
    return np.array(X_cdolg_test), np.array( y_cdolg_test)