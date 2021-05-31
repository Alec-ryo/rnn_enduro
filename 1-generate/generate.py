import numpy as np
import gym
import time
import os
from PIL import Image
from scipy.io import savemat
import random

def prepare_folder(PATH):
    m = []
    listdir = os.listdir(PATH)

    if listdir:
        for folder in listdir:
            m.append(int(folder.split("_")[1]))
        number_of_last_folder = max(m)
        next_folder = "match_" + str(number_of_last_folder + 1)
    else:
        next_folder = "match_0"

    match_path = PATH + next_folder
    os.mkdir(match_path)
    
    print(next_folder)
    
    img_path = PATH + next_folder + "/" + "img"
    os.mkdir(img_path)
    
    npz_path = PATH + next_folder + "/" + "npz"
    os.mkdir(npz_path)
    
    mat_path = PATH + next_folder + "/" + "mat"
    os.mkdir(mat_path)
    
    return match_path+"/", img_path+"/", npz_path+"/", mat_path+"/" 

def save_as_png(PATH, pil_buffer):

    print("salva png")

    _path = PATH

    for i in range(len(pil_buffer)):
        
        path = _path + str(i) + ".png"
        pil_buffer[i].save(path)

    print("Successfully saved as PNG.")

def save_as_npz(PATH):
    path = PATH

    np.savez_compressed(path + "frames.npz", match_buffer[1])
    np.savez_compressed(path + "actions.npz", match_buffer[2])
    np.savez_compressed(path + "rewards.npz", match_buffer[3])
    np.savez_compressed(path + "lifes.npz", match_buffer[4])

    print("Successfully saved as NPZ.")

def save_actions_as_txt(PATH, actions):
    f = open(PATH + "actions.txt", "w")
    for i in range(len(actions)):
        f.write(str(actions[i]) + " \n")
    f.close()

def save_as_matfile(PATH, match_buffer):
    mdic = {"num_frames": match_buffer[0], "frames": match_buffer[1], "actions": match_buffer[2]}
    savemat(PATH + "data.mat", mdic)

