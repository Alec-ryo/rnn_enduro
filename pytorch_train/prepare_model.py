from numpy.core.fromnumeric import choose
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import os

class prepare_model():
    def __init__(self, data):
        self.type = self.set_model_type()
        self.obs = input("observation: ")
        self.n_epochs = int(input("number of epochs: "))
        self.hidden_neurons = int(input("number of hidden neurons: "))
        self.device = self.set_gpu()
        self.model_name, self.model_path = self.set_model_path(data)
        self.min_loss = None
        self.criterion = None
        self.optimizer = None
    
    def get_info(self):
        print("type:", self.type)
        print("obs:", self.obs)
        print("n_epochs:", self.n_epochs)
        print("hidden_neurons:", self.hidden_neurons)
        print("device:", self.device)
        print("model_name:", self.model_name)
        print("min_loss", self.min_loss)
        print("criterion", self.criterion)
        print("optimizer", self.optimizer)
    
    def set_gpu(self):
        use_gpu = input("Use GPU (y/n) ")

        if use_gpu == 'y':
            use_gpu = True
        else:
            use_gpu = False
            
        device = self.conf_cuda(use_gpu)

        return device

    def conf_cuda(self, use_gpu):
    
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("Selected CPU")

        return device

    def set_model_type(self):
        print("1 - RNN")
        print("2 - LSTM")
        print("3 - CNN")

        choosed_type = int(input("Choose type of model: "))

        if choosed_type == 1:
            print("RNN model")
            choosed_type = "RNN"
        elif choosed_type == 2:
            print("LSTM model")
            choosed_type = "LSTM"
        elif choosed_type == 3:
            print("CNN model")
            choosed_type = "CNN"
        else:
            print("ERROR! not exists this type of model")
        
        return choosed_type

    def set_model_path(self, data):
        model_name = f"{self.obs}_m{data.start_match}to{data.end_match}_f{data.start_frame}to{data.end_frame}_epoch{self.n_epochs}_H{self.hidden_neurons}"
        newpath = f"models/" + model_name
        if not os.path.exists(newpath):
            print(f"models/" + model_name + " created")
            os.makedirs(newpath)
        else:
            print(f"models/" + model_name)
            print("ATTENTION! folder not created. Training informations will overwrite the existing one")
        return model_name, newpath

    def get_acc(self, predicted, target):
    
        predicted = torch.argmax(predicted, axis=1)
        target = torch.argmax(target, axis=1)
        
        correct = torch.sum(predicted == target)
        
        acc = correct/predicted.shape[0]

        return float(acc)

    def train_def(self, model):
        self.min_loss = 1e-5
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters())

class RNNModel(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_dim, n_layers, is_softmax):
        super(RNNModel, self).__init__()
        
        self.device = device

        # Defining some parameters
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.init_hidden()

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)  
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        if is_softmax:
            self.out = nn.Softmax()
        else:
            self.out = nn.Sigmoid()
    
    def forward(self, x):
        
        hidden = torch.randn(1, self.input_size, self.hidden_dim)
        
        pad_embed_pack_lstm = self.rnn(x, hidden)
        pad_embed_pack_lstm_pad = pad_packed_sequence(pad_embed_pack_lstm[0], batch_first=True)
        
        outs, _ = pad_embed_pack_lstm_pad
        
        out = outs.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        out = self.out(out)
        
        return out
        
    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(1, self.input_size, self.hidden_dim)
        hidden_b = torch.randn(1, self.input_size, self.hidden_dim)

        if self.device.type == 'cuda':
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

class LSTMModel(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_dim, n_layers, is_softmax):
        super(LSTMModel, self).__init__()
        
        self.device = device

        # Defining some parameters
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.init_hidden()

        #Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)  
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        if is_softmax:
            self.out = nn.Softmax()
        else:
            self.out = nn.Sigmoid()
    
    def forward(self, x):
        
        hidden = self.init_hidden()
        
        pad_embed_pack_lstm = self.lstm(x, hidden)
        pad_embed_pack_lstm_pad = pad_packed_sequence(pad_embed_pack_lstm[0], batch_first=True)
        
        outs, _ = pad_embed_pack_lstm_pad
        
        out = outs.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        out = self.out(out)
        
        return out
        
    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(1, self.input_size, self.hidden_dim)
        hidden_b = torch.randn(1, self.input_size, self.hidden_dim)

        if self.device.type == 'cuda':
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 500, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(500, 200, 5)
        self.fc1 = nn.Linear(200*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 200*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)

        return x
    