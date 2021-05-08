import numpy as np
import torch
import torch.nn as nn

def get_actions_list(zigzag=False):
    
    if zigzag:
        
        ACTIONS = {
            "right": 2,
            "left": 3,
        }

    else:

        ACTIONS = {
            "noop": 0,
            "accelerate": 1,
            "right": 2,
            "left": 3,
            "break": 4,
            "right_break": 5,
            "left_break": 6,
            "right_accelerate": 7,
            "left_accelerate": 8,
        }

    return list(ACTIONS.values())

def load_npz(data_path, m):
    
    path = data_path + "match_" + str(m) + "/npz/"

    actions = np.load(path + 'actions.npz')
    lifes = np.load(path + 'lifes.npz')
    frames = np.load(path + 'frames.npz')
    rewards = np.load(path + 'rewards.npz')

    arr_actions = actions.f.arr_0
    arr_lifes = lifes.f.arr_0
    arr_frames = frames.f.arr_0
    arr_rewards = rewards.f.arr_0

    print("Successfully loaded NPZ.")

    return arr_actions.shape[0], arr_frames, arr_actions, arr_rewards, arr_lifes

def prepare_action_data(action, ACTIONS_LIST):

    new_action = np.zeros((1, len(ACTIONS_LIST)), dtype=int) 

    new_action[0, ACTIONS_LIST.index(action)] = 1

    return new_action

class Model(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_dim, n_layers, is_softmax):
        super(Model, self).__init__()
        
        self.device = device

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # self.h0 = torch.zeros(self.n_layers, 1, self.hidden_dim).to(self.device)
        # self.c0 = torch.zeros(self.n_layers, 1, self.hidden_dim).to(self.device)

        self.init_hidden(1)

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
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        # hidden = self.init_hidden(batch_size)
        # self.h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        # self.c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)

        self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(x, (self.h0, self.c0))
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        out = self.out(out)
        
        return out, hidden

    def init_hidden(self, batch_size):

        if self.device.type == 'cuda':
            self.h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
            self.c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        else:
            self.h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
            self.c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
    
def conf_cuda(use_cuda):
    
    if use_cuda:
        
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
        is_cuda = torch.cuda.is_available()

        # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
        if is_cuda:
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")
    else:
        device = torch.device("cpu")
        print("Selected CPU")
    return device

def get_acc(predicted, target):
    
    predicted = torch.argmax(predicted, axis=1)
    target = torch.argmax(target, axis=1)
    
    correct = torch.sum(predicted == target)
    
    acc = correct/predicted.shape[0]
    return float(acc)