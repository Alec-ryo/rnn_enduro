import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel, self).__init__()
        
        self.device = device
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.init_hidden()
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.out = nn.Softmax()
    
    def forward(self, x):
        
        hidden = self.init_hidden()
        pad_embed_pack_lstm = self.rnn(x, hidden)

        pad_embed_pack_lstm_pad = pad_packed_sequence(pad_embed_pack_lstm[0], batch_first=True)
        
        outs, _ = pad_embed_pack_lstm_pad
        out = outs.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        out = self.out(out)
        
        return out
        
    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden = torch.randn(1, self.input_size, self.hidden_dim)

        if self.device.type == 'cuda':
            hidden = hidden.cuda()

        hidden = Variable(hidden)

        return hidden

class LSTMModel(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_dim, n_layers):
        super(LSTMModel, self).__init__()
        
        self.device = device

        # Defining some parameters
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.init_hidden()
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)  
        self.fc = nn.Linear(hidden_dim, output_size)
        
        self.out = nn.Softmax()
    
    def forward(self, x):
        
        hidden = self.init_hidden()

        pad_embed_pack_lstm = self.lstm(x, hidden)
        pad_embed_pack_lstm_pad = pad_packed_sequence(pad_embed_pack_lstm[0], batch_first=True)
        
        outs, _ = pad_embed_pack_lstm_pad
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
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

class CNNLSTMModel(nn.Module):
    def __init__(self, device, input_shape, output_size, hidden_dim, n_layers):
        super(CNNLSTMModel, self).__init__()
        
        self.device = device
        self.lstm_input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(3, 3, (10,10))
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(num_features=3)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

        # output = (input - filter + 1) / stride
        # convolução 1: (100 - 3 + 1) / 1 = 98x98
        # pooling 1: 49x49
        # convolução 2: (45 - 10 + 1) / 1 = 36x36
        # pooling 2: 18x18
        # 18 * 18 * 3
        self.init_hidden(49*49*3)
        self.lstm = nn.LSTM(49*49*3, hidden_dim, n_layers, batch_first=True)  
        self.fc = nn.Linear(hidden_dim, output_size)
        self.out = nn.Softmax()
    
    def forward(self, x):
        
        hidden = self.init_hidden(49*49*3)

        x.padSequence()
        original_shape = x.data.shape
        x.data = x.data.reshape(-1, 1, 100, 100)

        x = self.pool(self.bnorm(self.activation(self.conv1(x))))
        # x = self.pool(self.bnorm(self.activation(self.conv2(x))))

        x = x.reshape(original_shape[0], original_shape[1], -1)
        
        x = pack_padded_sequence(x, data.seq_len, batch_first=True, enforce_sorted=False)
        
        pad_embed_pack_lstm = self.lstm(x, hidden)
        pad_embed_pack_lstm_pad = pad_packed_sequence(pad_embed_pack_lstm[0], batch_first=True)
        
        outs, _ = pad_embed_pack_lstm_pad
        
        out = outs.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.out(out)
        
        return out
        
    def init_hidden(self, input_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(1, input_size, self.hidden_dim)
        hidden_b = torch.randn(1, input_size, self.hidden_dim)

        if self.device.type == 'cuda':
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)