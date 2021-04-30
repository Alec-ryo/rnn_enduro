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

use_gpu = input("Use GPU (y/n) ")
if use_gpu == 'y':
    use_gpu = True
else:
    use_gpu = False

device = conf_cuda(use_gpu)

if use_gpu:
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
num_of_frames_arr = np.array(num_of_frames_arr)

X_train = torch.tensor(X_train).float()
Y_train = torch.tensor(Y_train).float()

model = Model(device=device, input_size=20400, output_size=len(ACTIONS_LIST), hidden_dim=hidden_neurons, n_layers=1)

# We'll also set the model to the device that we defined earlier (default is CPU)
if use_gpu:
    model.cuda()
    X_train = X_train.cuda() 
    Y_train = Y_train.cuda()

min_loss = 1e-05
# Define Loss, Optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters())

train_loss_arr = np.array([])
train_acc_arr = np.array([])
valid_loss_arr = np.array([])
valid_acc_arr = np.array([])
valid_loss_mean_arr = np.array([])
valid_acc_mean_arr = np.array([])

# Training Run
loss_file = open(newpath + '/' + "loss_file.txt", "w")
first_time = True

best_loss = 1
first_epoch = True

start_time_processing = time.time()
for epoch in range(1, n_epochs + 1):

    model.train()

    optimizer.zero_grad() # Clears existing gradients from previous epoch
    X_train.to(device)
    output, hidden = model(X_train)
    loss = criterion(output, Y_train.view(-1,len(ACTIONS_LIST)).float())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordinglyw
        
    if epoch%10 == 0:

        train_loss_arr = np.append(train_loss_arr, loss.item())
        train_acc_arr  = np.append(train_acc_arr, get_acc(output, Y_train.reshape(-1, len(ACTIONS_LIST))))
    
        model.eval()
        
        epoch_valid_losses = np.array([])
        epoch_valid_acc = np.array([])
        for seq in range(len(X_train)):
            output, hidden = model(torch.unsqueeze(X_train[seq], 1))
            loss = criterion(output, Y_train[seq].view(-1,len(ACTIONS_LIST)).float())
            epoch_valid_losses = np.append(epoch_valid_losses, loss.item())
            epoch_valid_acc = np.append( epoch_valid_acc, get_acc(output, Y_train[seq].reshape(-1, len(ACTIONS_LIST))) )
            
        if first_epoch:
            valid_loss_arr = epoch_valid_losses.reshape(-1, 1)
            valid_acc_arr = epoch_valid_acc.reshape(-1, 1)
            first_epoch = False
        else:
            valid_loss_arr = np.insert(valid_loss_arr, valid_loss_arr.shape[1], epoch_valid_losses, axis=1)
            valid_acc_arr = np.insert(valid_acc_arr, valid_acc_arr.shape[1], epoch_valid_acc, axis=1)
            
        valid_loss_mean_arr = np.append(valid_loss_mean_arr, np.mean(epoch_valid_losses))
        valid_acc_mean_arr = np.append(valid_acc_mean_arr, np.mean(epoch_valid_acc))
        
        loss_file.write("Epoch: {}/{}-------------------------------------------\n".format(epoch, n_epochs))
        loss_file.write("Train -> Loss: {:.15f} Acc: {:.15f}\n".format(train_loss_arr[-1], train_acc_arr[-1]))
        loss_file.write("Valid -> Loss: {:.15f} Acc: {:.15f}\n".format(valid_loss_mean_arr[-1], valid_acc_mean_arr[-1]))
            
        print("Epoch: {}/{}-------------------------------------------".format(epoch, n_epochs))
        print("Train -> Loss: {:.15f} Acc: {:.15f}".format(train_loss_arr[-1], train_acc_arr[-1]))
        print("Valid -> Loss: {:.15f} Acc: {:.15f}".format(valid_loss_mean_arr[-1], valid_acc_mean_arr[-1]))
        
        if train_loss_arr[-1] < best_loss:
            state = { 'epoch': epoch + 1, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'losslogger': loss.item(), }
            torch.save(state, newpath + '/' + model_name)
            best_loss = loss.item()
        else:
            print("model not saved")
        
        if (valid_loss_mean_arr[-1] < min_loss):
            break
        
loss_file.close()
np.savez(newpath + '/' + "train_loss_arr", train_loss_arr)
np.savez(newpath + '/' + "valid_loss_arr", valid_loss_arr)
np.savez(newpath + '/' + "valid_loss_mean_arr", valid_loss_mean_arr)
print("--- %s seconds ---" % (time.time() - start_time_processing))

# summarize history for loss
plt.plot(train_loss_arr, color='blue')
plt.title('model train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.yscale('log')
plt.savefig(newpath + '/' + 'train_loss.png')

# summarize history for loss
plt.plot(valid_loss_mean_arr, color='blue')
plt.title('model valid loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['valid'], loc='upper left')
plt.yscale('log')
plt.savefig(newpath + '/' + 'valid_loss_mean.png')

for seq in range(len(X_train)):
    # summarize history for loss
    plt.plot(valid_loss_arr[seq], color='blue')
    plt.title('model valid loss ' + str(seq))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['valid'], loc='upper left')
    plt.yscale('log')
    plt.savefig(newpath + '/' + f'valid_loss_{seq}.png')
