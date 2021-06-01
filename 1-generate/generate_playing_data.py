import numpy as np
import gym
import time
import os
from PIL import Image
from scipy.io import savemat
import keyboard

sleep_time = 0.05
path = "data/"

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

def control():

    command = 0
    # Se o gatilho estiver apertado
    if keyboard.is_pressed('s'):
        if keyboard.is_pressed('d') and not(keyboard.is_pressed('a')):
            command = 5
        elif keyboard.is_pressed('a') and not(keyboard.is_pressed('d')):
            command = 6
        else:
            command = 4

    elif(not(keyboard.is_pressed('s'))):
        if keyboard.is_pressed('space'):
            if keyboard.is_pressed('a') and not(keyboard.is_pressed('d')):
                command = 8
            elif keyboard.is_pressed('d') and not(keyboard.is_pressed('a')):
                command = 7
            else:
                command = 1
        elif not(keyboard.is_pressed('space')):
            if keyboard.is_pressed('d') and not(keyboard.is_pressed('a')):
                    command = 2
            elif keyboard.is_pressed('a') and not(keyboard.is_pressed('d')):
                command = 3
            else:
                command = 0
    
    return command

y_min, y_max, x_min, x_max = 25, 195, 20, 140
shape_of_single_frame = (1, (y_max-y_min),(x_max-x_min))

match_buffer = []
pil_buffer = []
frame_buffer = np.empty((0, y_max-y_min, x_max-x_min), dtype=np.double)
action_buffer = np.empty((0,1), dtype=int) 
reward_buffer = np.empty((0,1), dtype=np.double)
life_buffer = np.empty((0,1), dtype=np.double)

num_frames = int(input("Quantidade de frames a ser gerada: "))

env = gym.make("Enduro-v0")
frame = env.reset()
reward, action, done, info = 0, 0, False, {'ale.lives': 0}

for i in range(num_frames):

    if keyboard.is_pressed('p'):
        time.sleep(0.01)
        while(not(keyboard.is_pressed('p'))):
            pass
    
    print(i)
    if i < num_frames/2:
        time.sleep(sleep_time)
    env.render()
    action = control()

    frame = frame[y_min:y_max, x_min:x_max]

    frame = Image.fromarray(frame)
    frame = frame.convert("L")
    
    pil_buffer.append(frame)
    
    frame = np.asarray(frame)
    frame = frame.reshape(1, y_max-y_min, x_max-x_min)

    frame_buffer = np.concatenate(( frame_buffer, frame ))
    reward_buffer = np.append(reward_buffer, np.array(reward))
    life_buffer = np.append(life_buffer, np.array(info['ale.lives']))
    action_buffer = np.append(action_buffer, np.array(action))

    frame, reward, done, info = env.step(action)

match_path, img_path, npz_path, mat_path = prepare_folder(path)

match_buffer = [action_buffer.shape[0], frame_buffer, action_buffer, reward_buffer, life_buffer]

save_as_npz(npz_path)
save_as_png(img_path, pil_buffer)
save_actions_as_txt(mat_path, action_buffer)
save_as_matfile(mat_path, match_buffer)