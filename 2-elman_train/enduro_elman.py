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
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        out = self.sigmoid(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
def conf_cuda(use_cuda=True):
    
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