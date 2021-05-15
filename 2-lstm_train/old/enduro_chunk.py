
indices = [[  0, 52, 62, 80, 106, 117, 137, 142, 156, 159, 166, 185, 215, 226, 235, 242, 252, 259, 279, 281, 292, 296, 299, 
                303, 308, 315, 318, 321, 329, 333, 335, 354, 360, 374, 377, 382, 386, 390, 393, 402, 414, 420, 444, 447, 
                465, 469, 487, 491, 504, 506, 510, 513, 525, 528, 538, 552, 558, 570, 576, 581, 584, 588, 595, 598, 604, 
                607, 610, 618, 619, 622, 631, 641, 645, 662, 672, 685, 693, 697, 704, 718, 724, 726, 735, 739, 745, 754, 
                755, 770, 773, 789, 790, 799, 805, 809, 823, 834, 850, 854, 857, 868, 874, 875, 877, 883, 885, 904, 906, 
                911, 919, 923, 926, 932, 943, 962, 965, 984, 991, 999, 1007, 1010, 1015, 1017, 1018],
           [   0, 31, 45, 47, 56, 63, 69, 85, 88, 92, 96, 112, 118, 124, 130, 137, 143, 151, 158, 170, 184, 202, 223, 243, 249,
             255,260,269,272,275,276,282,285,295,296,297,306,308,318,322,331,334,337,338,341,343,354,357,381,387,400,402,408,
             420,423,426,429,441,443,445,448,454,463,465,474,481,502,506,524,533,538,548,556,562,566,570,581,588,603,604,609,
             652,668,681,698,710,714,725,729,736,739,743,748,749,752,766,770,774,778,781,792,795,803,804,806,814,818,821,828,
             834,837,843,849,855,862,868,874,884,888,895,896,919,935,942,945,957,958,963,964,965,977,980,984,989,997,1000],
           [   0,37,53,81,94,99,119,123,124,142,144,146,162,193,211,213,222,230,232,235,238,240,252,255,258,267,270,275,
                278,284,287,289,300,307,313,331,352,359,371,379,386,389,392,409,412,432,435,440,442,446,462,467,472,
                506,527,531,535,553,557,561,571,579,589,602,607,609,612,616,621,631,634,638,640,644,645,661,669,673,676,
                692,695,698,701,711,717,720,721,734,744,747,750,755,759,774,785,793,807,859,891,895,912,927,948,958,969,
                980,997,998,1004],
           [   0,33,58,75,86,100,118,120,139,142,147,159,162,165,167,174,179,180,183,192,197,199,209,214,217,222,227,231,
                232,239,244,251,262,266,271,275,277,282,287,291,294,297,308,311,316,318,322,327,339,342,348,352,362,372,
                383,397,399,402,405,417,420, 425,428,454,457,464,470,477,481,485,488,494,495,503,507,517,535,537,540,553,
                556,558,577,588,598,601,609,620,634,638,647,650,656,659,660,663,669,672,681,684,696,702,705,708,716,719,726,
                730,739,743,746,758,765,766,777,783,785,786,790,792,800,802,808,809,810,816,819,823,827,835,837,842,845,853,
                865,868,873,885,889,891,897,904,906,918,941,949,956,961,969,973,976,981,984,991,996,999,1002],
           [   0,59,68,73,76,86,94,97,101,104,109,118,120,123,128,134,136,139,143,145,148,154,160,162,177,190,195,199,204,
                208,212,220,221,222,232,234,246,251,253,256,263,270,273,287,302,314,322,355,357,368,396,397,401,416,428,433,
                437,442,446,452,456,469,475,490,500,506,525,526,538,545,551,554,558,562,571,579,582,586,587,600,612,664,823,
                824,832,850,858,870,879,881,889,898,901,913,916,924,932,936,943,952,955,975,985,992,997,1001],
           [   0,36,56,94,114,121,136,140,143,153,159,162,171,174,182,190,197,201,215,219,237,239,249,251,254,265,269,297,322,
                335,340,343,348,352,355,363,368,371,375,384,388,395,405,411,439,461,464,468,477,483,485,487,494,497,501,503,
                514,519,521,526,529,532,538,547,551,555,560,579,584,597,600,602,607,609,617,623,626,634,638,644,649,652,663,
                671,672,675,678,685,688,699,700,712,719,724,726,735,744,753,756,775,797,806,812,818,826,829,832,848,852,856,
                866,869,872,875,881,886,891,893,895,901,910,930,936,940,953,958,989,997,1004,1006] 
          ]

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

obs = input('escreva uma observacao (sem espaco): ')

if obs == 'zigzag':
    zigzag = True
else:
    zigzag = False

data_path = r"../1-generate/data/"
n_epochs = int(input("number of epochs: ") ) #5000
hidden_neurons = int(input("number of hidden neurons: ")) #500
stop_train = 1e-5

start_match = int(input("start match: ")) #45
end_match = int(input("end match: ")) #49

start_frame = int(input("start frame: ")) #1
end_frame = int(input("end frame: ")) #1000

is_softmax = True

model_name = f"{obs}_m{start_match}to{end_match}_f{start_frame}to{end_frame}_epoch{n_epochs}_H{hidden_neurons}"
newpath = f"models/" + model_name
if not os.path.exists(newpath):
    print(f"models/" + model_name + " created")
    os.makedirs(newpath)
else:
    print(f"models/" + model_name)
    print("ATTENTION! folder not created. Training informations will overwrite the existing one")

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

data_chunked = []
for i in range(len(frames_arr)):
    for j in range(len(indices[i]) - 1):
        data_chunked.append(frames_arr[i][indices[i][j]:indices[i][j+1]])

X_train = np.array(data_chunked)/255
Y_train = np.array(actions_arr)
num_of_frames_arr = np.array(num_of_frames_arr)

X_train = list(map(lambda x: torch.tensor(x), X_train))
Y_train = list(map(lambda x: torch.tensor(x), Y_train))

seq_len = torch.FloatTensor(list(map(len,X_train)))

X_train = pad_sequence(X_train, batch_first=True).float()
Y_train = pad_sequence(Y_train, batch_first=True).float()

X_train = pack_padded_sequence(X_train, seq_len, batch_first=True, enforce_sorted=False)

model = Model(device=device, input_size=12000, output_size=len(ACTIONS_LIST), hidden_dim=hidden_neurons, n_layers=1)

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

all_idx = np.arange(len(data))
start_time_processing = time.time()

# Training Run
loss_file = open(newpath + '/' + "loss_file.txt", "w")
first_time = True

best_loss = 1
first_epoch = True

for epoch in range(1, n_epochs + 1):

    model.train()

    optimizer.zero_grad() # Clears existing gradients from previous epoch
    packed.to(device)
    output = model(packed)
    loss = criterion(output, target_padded.view(-1,len(ACTIONS_LIST)).float())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordinglyw
        
    if epoch%10 == 0:

        train_loss_arr = np.append(train_loss_arr, loss.item())
        #train_acc_arr  = np.append(train_acc_arr, get_acc(output, target_padded.reshape(-1, len(ACTIONS_LIST))))
        train_acc_arr  = np.append(train_acc_arr, get_acc_2(output, target_padded.reshape(-1, len(ACTIONS_LIST))))
        
        loss_file.write("Epoch: {}/{}-------------------------------------------\n".format(epoch, n_epochs))
        loss_file.write("Train -> Loss: {:.15f} Acc: {:.15f}\n".format(train_loss_arr[-1], train_acc_arr[-1]))
            
        print("Epoch: {}/{}-------------------------------------------".format(epoch, n_epochs))
        print("Train -> Loss: {:.15f} Acc: {:.15f}".format(train_loss_arr[-1], train_acc_arr[-1]))
        
        if train_loss_arr[-1] < best_loss:
            state = { 'epoch': epoch + 1, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'losslogger': loss.item(), }
            torch.save(state, newpath + '/' + model_name)
            best_loss = loss.item()
        else:
            print("model not saved")
            
loss_file.write("--- %s seconds ---" % (time.time() - start_time_processing))
loss_file.close()
np.savez(newpath + '/' + "train_loss_arr", train_loss_arr)
#np.savez(newpath + '/' + "valid_acc_table", valid_loss_mean_arr)
print("--- %s seconds ---" % (time.time() - start_time_processing))

# summarize history for loss
plt.clf()
plt.plot(train_loss_arr, color='blue')
plt.title('model train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.yscale('log')
plt.savefig(newpath + '/' + 'train_loss.png')