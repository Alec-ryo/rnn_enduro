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
    

data = np.array(frames_arr)/255
targets = np.array(actions_arr)
num_of_frames = np.array(num_of_frames_arr)

data = torch.tensor(data).float()
targets = torch.tensor(targets).float()

all_idx = np.arange(len(data))

model = Model(device=device, input_size=20400, output_size=len(ACTIONS_LIST), hidden_dim=hidden_neurons, n_layers=1)

# We'll also set the model to the device that we defined earlier (default is CPU)
if use_gpu:
    model.cuda()
    data = data.cuda() 
    targets = targets.cuda()

min_loss = 1e-05
# Define Loss, Optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters())

train_loss_arr = np.array([])
train_acc_arr = np.array([])
test_loss_arr = np.array([])
test_acc_arr = np.array([])

# Training Run

epoch = 1
loss_file = open(newpath + '/' + "loss_file.txt", "w")
first_time = True

best_loss = 1

start_time_processing = time.time()
for epoch in range(1, n_epochs + 1):

    model.train()

    train_idx = np.random.choice(len(data), len(data) - 1, replace=False)
    test_idx = np.setdiff1d(all_idx, train_idx)

    X_train = data[train_idx]
    Y_train = targets[train_idx]
    X_test = data[test_idx]
    Y_test = targets[test_idx]

    optimizer.zero_grad() # Clears existing gradients from previous epoch
    X_train.to(device)
    output, hidden = model(X_train)
    loss = criterion(output, Y_train.view(-1,len(ACTIONS_LIST)).float())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordinglyw
        
    if epoch%10 == 0:

        train_loss = loss.item()
        train_acc = float(torch.sum((torch.argmax(output, axis=1) == torch.argmax(Y_train.reshape(-1, len(ACTIONS_LIST)), axis=1).int())/num_of_frames.sum()))

        model.eval()
        
        output, hidden = model(X_test)
        loss = criterion(output, Y_test.view(-1,len(ACTIONS_LIST)).float())
        
        valid_loss = loss.item()    
        valid_acc = float(torch.sum((torch.argmax(output, axis=1) == torch.argmax(Y_test.reshape(-1, len(ACTIONS_LIST)), axis=1).int())/num_of_frames.sum()))
        
        loss_file.write("Epoch: {}/{}-------------------------------------------\n".format(epoch, n_epochs))
        loss_file.write("Train    -> Loss: {:.15f} Acc: {:.15f}\n".format(train_loss, train_acc))
        loss_file.write("Valid{} -> Loss: {:.15f} Acc: {:.15f}\n".format(test_idx, valid_loss, valid_acc))
        
        print('Epoch: {}/{}-------------------------------------------'.format(epoch, n_epochs))
        print("Train    -> Loss: {:.15f} Acc: {:.15f}".format(train_loss, train_acc))
        print("Valid{} -> Loss: {:.15f} Acc: {:.15f}\n".format(test_idx, valid_loss, valid_acc))
        
        if train_loss < best_loss:
            state = { 'epoch': epoch + 1, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'losslogger': loss.item(), }
            torch.save(state, newpath + '/' + model_name)
            best_loss = loss.item()

        train_loss_arr = np.append(train_loss_arr, train_loss)
        train_acc_arr = np.append(train_acc_arr, train_acc)
        test_loss_arr = np.append(test_loss_arr, valid_loss)
        test_acc_arr = np.append(test_acc_arr, valid_acc)
        
        if (valid_loss < min_loss):
            break
        
loss_file.close()
np.savez(newpath + '/' + "train_loss_arr", train_loss_arr)
np.savez(newpath + '/' + "test_loss_arr", test_loss_arr)
print("--- %s seconds ---" % (time.time() - start_time_processing))

# summarize history for loss
plt.plot(train_loss_arr, color='blue')
plt.plot(test_loss_arr, color='red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.yscale('log')
plt.savefig(newpath + '/' + 'loss.png')
plt.show()
