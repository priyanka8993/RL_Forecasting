import pandas as pd 
import numpy as np

class preprocess():
    def __init__(self,input_tw,output_tw):
        self.input_tw = input_tw
        self.output_tw = output_tw

    def get_train_seq(self):
        train_data = pd.read_csv('C:/Users/priyanka.chakraborty/Projects/COE/RL/Projects/Fastenal_v2/Fastenal/data/Store_Product_Quantity_Train.csv', sep = '\t')
        print(train_data.head())
        train_x_seq, train_y_seq = self.get_data_seq(train_data)
        print(train_x_seq.shape)
        print(train_y_seq.shape)
        return train_x_seq, train_y_seq
    
    def get_test_seq(self):
        test_data = pd.read_csv('C:/Users/priyanka.chakraborty/Projects/COE/RL/Projects/Fastenal_v2/Fastenal/data/Store_Product_Quantity_Test.csv', sep = '\t')
        test_x_seq, test_y_seq = self.get_data_seq(test_data)
        print(test_x_seq.shape)
        print(test_y_seq.shape)
        return test_x_seq, test_y_seq

    def get_data_seq(self, data):
        preprocess_data = self.all_seq(data)
        for product_id in preprocess_data:
            X_seq_array, y_seq_array = self.create_seq_array(preprocess_data,product_id)
        return X_seq_array, y_seq_array
        
       
    def create_sequences(self,data):
        inout_seq = []
        L = len(data)
        for i in range(L-self.input_tw):
            _seq = data[i:i+self.input_tw]
            _label = data[i+self.input_tw:i+self.input_tw+self.output_tw]
            inout_seq.append((_seq ,_label))
        return inout_seq

    def all_seq(self,data):
        train_data_grouped = data.groupby('INV_ITEM_ID')
        feature_sequences = {}
        for i in train_data_grouped:
            sequences = self.create_sequences(i[1])
            n_seq=len(i[1]) - self.input_tw- self.output_tw + 1
            feature_sequences[i[0]] = sequences[:n_seq]
        return feature_sequences

    def create_seq_array(self,feature_sequences,prodID):
        X = []
        y = []
        for seq, out in feature_sequences[prodID]:
            X.append(seq['QTY_CONSUMED']);y.append(out['QTY_CONSUMED'])   
        
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y