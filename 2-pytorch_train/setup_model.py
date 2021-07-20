import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import os
from setup_model_types import *

class ModelStructure():

    def __init__(self, device, data_size, output_size, match_list, start_frame, end_frame):

        self.device = device
        self.data_size = data_size
        self.output_size = output_size
        self.match_list = match_list
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.hidden_neurons = int(input("Number of hidden neurons: "))
        self.n_epochs = int(input("Number of epochs: "))
        self.min_loss = 1e-5
        self.n_layers = 1
        self.type = self.set_type_name()
        self.name = self.setName()
        self.path = self.setPath()

    def set_type_name(self):
        print("Choose type of RNN model:")
        print("1 - Simple RNN")
        print("2 - LSTM")
        print("3 - CNN")
        choise = int(input("type: "))
        if choise == 1:
            return "RNN"
        elif choise == 2:
            return "LSTM"
        elif choise == 3:
            return "CNN"
        else:
            return "Error"

    def setName(self):
        x = [str(num) for num in self.match_list]
        x = '-'.join(x)
        obs = input("write a observations without space and punctuations:")
        name = f"{self.type}_{obs}_m{x}_f{self.start_frame}to{self.end_frame}_epoch{self.n_epochs}_h{self.hidden_neurons}"
        return name

    def setPath(self):
        newpath = f"models/" + self.name
        if not os.path.exists(newpath):
            print(f"models/" + self.name + " created")
            os.makedirs(newpath)
        else:
            print(f"models/" + self.name)
            print("ATTENTION! folder not created. Training informations will overwrite the existing one")
        return newpath
        
    def get_acc(self, predicted, target):
    
        predicted = torch.argmax(predicted, axis=1)
        target = torch.argmax(target, axis=1)

        correct = torch.sum(predicted == target)

        acc = correct/predicted.shape[0]
        return float(acc)