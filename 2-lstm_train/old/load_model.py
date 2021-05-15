import numpy as np
import torch
import torch.nn as nn
from enduro_lstm import *
import gym
import time
from PIL import Image

data_path = r"../1-generate/data/"
use_cuda = False
zigzag = False

newpath = 'models/play_m45to46_f1to1000_epoch10000_H500/' + 'play_m45to46_f1to1000_epoch10000_H500'

hidden_neurons = 500
n_output = 9

device = conf_cuda(use_cuda)

model = Model(device=device, input_size=20400, output_size=n_output, hidden_dim=hidden_neurons, n_layers=1)
model.load_state_dict(torch.load(newpath, map_location=device))
model.eval()

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

y_min, y_max, x_min, x_max = 25, 195, 20, 140
shape_of_single_frame = (1, (y_max-y_min),(x_max-x_min))

sleep_time = 0.05

env = gym.make("Enduro-v0")
frame = env.reset()
reward, action, done, info = 0, 0, False, {'ale.lives': 0}

hx = torch.zeros(1, hidden_neurons)
cx = torch.zeros(1, hidden_neurons)

while(True):
    
    time.sleep(sleep_time)
    env.render()
    
    frame = frame[y_min:y_max, x_min:x_max]

    frame = Image.fromarray(frame)
    frame = frame.convert("L")
    
    frame = np.asarray(frame)
    frame = frame.reshape(1, -1)
    frame = torch.tensor(frame)/255
    
    hx, cx = lstmcell(frame, (hx, cx))
    out = linear(hx)
    action = sigmoid(out)
    
    action = list(ACTIONS.values())[torch.argmax(action, axis=1)]
    frame, reward, done, info = env.step(action)