import numpy as np
import torch
import torch.nn as nn
import csv
import os
import cv2
from PIL import Image
from enduro_lstm import *
import matplotlib.pyplot as plt
import time

device = conf_cuda(True)

torch.cuda.empty_cache()

obs = 'play'
if obs == 'zigzag':
    zigzag = True
else:
    zigzag = False
zigzag

data_path = r"../1-generate/data/"
n_epochs = int(input("number of epochs: ") ) #5000
hidden_neurons = int(input("number of hidden neurons: ")) #500
stop_train = 1e-5

start_match = int(input("start match: ")) #45
end_match = int(input("end match: ")) #49

start_frame = int(input("start frame: ")) #1
end_frame = int(input("end frame: ")) #1000

model_name = f"{obs}_m{start_match}to{end_match}_f{start_frame}to{end_frame}_epoch{n_epochs}_H{hidden_neurons}"
newpath = f"models/" + model_name
if not os.path.exists(newpath):
    print(f"models/" + model_name + " created")
    os.makedirs(newpath)
else:
    print(f"models/" + model_name)
    trained = input("ATTENTION! folder not created. Training informations will overwrite the existing one, you want to continue (y/n)")
    if trained != ('y'):
        exit()

ACTIONS_LIST = get_actions_list(zigzag=zigzag)

num_of_frames_arr = []
frames_arr = []
actions_arr = []

for m in range(start_match, end_match + 1):
    
    num_of_frames, frames, actions, rewards, lifes = load_npz(data_path, m)
    frames = frames[start_frame - 1:end_frame]
    actions = actions[start_frame - 1:end_frame]
    
    action_one_hot = [prepare_action_data(i, ACTIONS_LIST) for i in actions]
    actions = np.array(action_one_hot)
    actions = actions.reshape(len(actions), -1)
    
    frames_arr.append(frames)
    actions_arr.append(actions)
    num_of_frames_arr.append(end_frame - start_frame + 1) 
    

X_train = np.array(frames_arr)/255
Y_train = np.array(actions_arr)
num_of_frames = np.array(num_of_frames_arr)

X_train = torch.tensor(X_train).float()
Y_train = torch.tensor(Y_train).float()

model = Model(device=device, input_size=20400, output_size=len(ACTIONS_LIST), hidden_dim=hidden_neurons, n_layers=1)

# We'll also set the model to the device that we defined earlier (default is CPU)
model.cuda()
X_train = X_train.cuda() 
Y_train = Y_train.cuda()

min_loss = 1e-05
# Define Loss, Optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters())

loss_arr = np.array([])
acc_arr = np.array([])

# Training Run

epoch = 1
loss_file = open(newpath + '/' + "loss_file.txt", "w")
first_time = True

best_loss = 0

start_time_processing = time.time()
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    X_train.to(device)
    output, hidden = model(X_train)
    loss = criterion(output, Y_train.view(-1,len(ACTIONS_LIST)).float())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordinglyw
    
    if epoch%10 == 0:
        acc = float(torch.sum((torch.argmax(output, axis=1) == torch.argmax(Y_train.reshape(-1, len(ACTIONS_LIST)), axis=1).int())/num_of_frames.sum()))
        
        loss_file.write("Epoch: {}/{}.............".format(epoch, n_epochs))
        loss_file.write("Loss: {:.15f} Acc: {:.15f}\n".format(loss.item(), acc))
        
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.15f} Acc: {:.15f}".format(loss.item(), acc))
        
        if best_loss < loss.item():
            state = { 'epoch': epoch + 1, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'losslogger': loss.item(), }
            torch.save(state, filename)
            best_loss = loss.item()

        loss_arr = np.append(loss_arr, loss.item())
        acc_arr = np.append(loss_arr, loss.item())
        
        if (acc_arr[-1] - acc > 0.2) or (loss.item() < min_loss):
            break
        
loss_file.close()
np.savez(newpath + '/' + "loss_arr", loss_arr)
print("--- %s seconds ---" % (time.time() - start_time_processing))

# summarize history for loss
g = list(loss_arr)
plt.plot(loss_arr)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.yscale('log')
plt.savefig(newpath + '/' + 'perf.png')
plt.show()

