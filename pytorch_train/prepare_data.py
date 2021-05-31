import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class prepare_data():

    def __init__(self):
        
        self.data_path = r"../1-generate/data/"
        self.start_match = None
        self.end_match = None

        self.start_frame = None
        self.end_frame = None

        self.actions_list = None
        self.train_data, self.targets = None, None

        self.data_shape = None
        self.target_shape = None

    def load_npz(self, data_path, m):
    
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
        
    def action_to_onehotencode(self, action, ACTIONS_LIST):

        new_action = np.zeros((1, len(ACTIONS_LIST)), dtype=int) 

        new_action[0, ACTIONS_LIST.index(action)] = 1

        return new_action

    def get_actions_list(self, zigzag=False):
    
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
    
    def get_entire_seq(self):
        
        frames_arr = []
        actions_arr = []

        for m in range(self.start_match, self.end_match + 1):
            
            _, frames, actions, _, _ = self.load_npz(self.data_path, m)
            frames = frames[self.start_frame - 1:self.end_frame, 30:130, 10:110]

            actions = actions[self.start_frame - 1:self.end_frame]
            action_one_hot = [self.action_to_onehotencode(i, self.actions_list) for i in actions]
            actions = np.array(action_one_hot)
            
            frames_arr.append(frames)
            actions_arr.append(actions)

        return np.array(frames_arr), np.array(actions_arr)

    def normalize_data(self, data):
        return data/255

    def print_data(self):
        print("data path:", self.data_path)

        print("start match:", self.start_match)
        print("end match:", self.end_match)

        print("start frame:", self.start_frame)
        print("end frame:", self.end_frame)

        print("available actions:", self.actions_list)
        print("train data shape:", self.train_data.shape)
        print("targets data shape:", self.targets.shape)

    def to_tensor(self, data):
        return list(map(lambda x: torch.tensor(x), data))

    def get_seqs_len(self, data):
        return torch.FloatTensor(list(map(len, data)))
    
    def get_pad_sequence(self, data):
        return pad_sequence(data, batch_first=True).float()
    
    def get_pack_sequence(self, data, seq_len):
        return pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=False)

    def pack_to_model(self, frames, actions):
        frames, actions = self.to_tensor(frames), self.to_tensor(actions)
        seqs_len = self.get_seqs_len(frames)
        frames, actions = self.get_pad_sequence(frames), self.get_pad_sequence(actions)
        frames = self.get_pack_sequence(frames, seqs_len)

        return frames, actions

class zigzag(prepare_data):

    def __init__(self):
        super().__init__()

        self.start_match = 0
        self.end_match = 0

        self.start_frame = 1
        self.end_frame = 120

        self.actions_list = self.get_actions_list(True)
        self.train_data, self.targets = self.get_data()

        self.data_shape = self.train_data.shape
        self.target_shape = self.targets.shape

    def get_data(self):
        frames, actions = self.get_entire_seq()
        frames = self.normalize_data(frames)
        return frames, actions

class entire_seq(prepare_data):

    def __init__(self):
        super().__init__()

        self.start_match = int(input("start match: "))
        self.end_match = int(input("end match: "))

        self.start_frame = int(input("start frame: "))
        self.end_frame = int(input("end frame: "))

        self.actions_list = self.get_actions_list(zigzag = False)
        self.train_data, self.targets = self.get_data()

        self.data_shape = self.train_data.shape
        self.target_shape = self.targets.shape

    def get_data(self):
        frames, actions = self.get_entire_seq()
        frames = self.normalize_data(frames)
        return frames, actions

class chuncked_seq(entire_seq):

    def __init__(self):
        super().__init__()

        self.list_idx_events = self.get_list_idx_events()
        self.train_data, self.targets = self.get_data()

        self.data_shape = self.train_data.shape
        self.target_shape = self.targets.shape

    def get_data(self):
        frames, actions = self.get_entire_seq()
        frames, actions = self.chunck_events(frames, actions)
        frames = self.normalize_data(frames)
        frames, actions = self.filter_seq(frames, actions, 100)
        
        return frames, actions 

    def chunck_events(self, frames, targets):
        data_chuncked = []
        target_chuncked = []

        n_matches = len(frames)

        for i in range(n_matches):
            n_match_events = len(self.list_idx_events[i])
            for j in range(n_match_events - 1):
                data_chuncked.append(frames[i][self.list_idx_events[i][j]:self.list_idx_events[i][j+1]])
                target_chuncked.append(targets[i][self.list_idx_events[i][j]:self.list_idx_events[i][j+1]])

        return np.array(data_chuncked), np.array(target_chuncked)

    def filter_seq(self, frames, targets, threshold):
        X_train_filtrado = []
        Y_train_filtrado = []
        for i in range(len(frames)):
            if len(frames[i]) > threshold:
                pass
            else:
                X_train_filtrado.append(frames[i])
                Y_train_filtrado.append(targets[i])
        frames = np.array(X_train_filtrado)
        targets = np.array(Y_train_filtrado)

        return frames, targets

    def get_list_idx_events():
        index   = [[  0,  62,  80, 106, 117, 137, 142, 156, 159, 166, 185, 215, 226, 235, 242, 252, 259, 279, 281, 292, 296, 299, 
                    303, 308, 315, 318, 321, 329, 333, 335, 354, 360, 374, 377, 382, 386, 390, 393, 402, 414, 420, 444, 447, 465, 
                    469, 487, 491, 504, 506, 510, 513, 525, 528, 538, 552, 558, 570, 576, 581, 584, 588, 595, 598, 604, 607, 610, 
                    618, 619, 622, 631, 641, 645, 662, 672, 685, 693, 697, 704, 718, 724, 726, 735, 739, 745, 754, 755, 770, 773, 
                    789, 790, 799, 805, 809, 823, 834, 850, 854, 857, 868, 874, 875, 877, 883, 885, 904, 906, 911, 919, 923, 926, 
                    932, 943, 962, 965, 984, 991, 999, 1007, 1010, 1015, 1017, 1018],
                   [  0,  31,  45,  47,  56,  63,  69,  85,  88,  92,  96, 112, 118, 124, 130, 137, 143, 151, 158, 170, 184, 202, 
                    223, 243, 249, 255, 260, 269, 272, 275, 276, 282, 285, 295, 296, 297, 306, 308, 318, 322, 331, 334, 337, 338, 
                    341, 343, 354, 357, 381, 387, 400, 402, 408, 420, 423, 426, 429, 441, 443, 445, 448, 454, 463, 465, 474, 481,
                    502, 506, 524, 533, 538, 548, 556, 562, 566, 570, 581, 588, 603, 604, 609, 652, 668, 681, 698, 710, 714, 725, 
                    729, 736, 739, 743, 748, 749, 752, 766, 770, 774, 778, 781, 792, 795, 803, 804, 806, 814, 818, 821, 828, 834, 
                    837, 843, 849, 855, 862, 868, 874, 884, 888, 895, 896, 919, 935, 942, 945, 957, 958, 963, 964, 965, 977, 980,
                    984, 989, 997, 1000], 
                   [  0,  37,  53,  81,  94,  99, 119, 123, 124, 142, 144, 146, 162, 193, 211, 213, 222, 230, 232, 235, 238, 240, 
                    252, 255, 258, 267, 270, 275, 278, 284, 287, 289, 300, 307, 313, 331, 352, 359, 371, 379, 386, 389, 392, 409, 
                    412, 432, 435, 440, 442, 446, 462, 467, 472, 506, 527, 531, 535, 553, 557, 561, 571, 579, 589, 602, 607, 609, 
                    612, 616, 621, 631, 634, 638, 640, 644, 645, 661, 669, 673, 676, 692, 695, 698, 701, 711, 717, 720, 721, 734, 
                    744, 747, 750, 755, 759, 774, 785, 793, 807, 859, 891, 895, 912, 927, 948, 958, 969, 980, 997, 998, 1004], 
                   [  0,  33,  58,  75,  86, 100, 118, 120, 139, 142, 147, 159, 162, 165, 167, 174, 179, 180, 183, 192, 197, 199, 
                    209, 214, 217, 222, 227, 231, 232, 239, 244, 251, 262, 266, 271, 275, 277, 282, 287, 291, 294, 297, 308, 311, 
                    316, 318, 322, 327, 339, 342, 348, 352, 362, 372, 383, 397, 399, 402, 405, 417, 420, 425, 428, 454, 457, 464, 
                    470, 477, 481, 485, 488, 494, 495, 503, 507, 517, 535, 537, 540, 553, 556, 558, 577, 588, 598, 601, 609, 620, 
                    634, 638, 647, 650, 656, 659, 660, 663, 669, 672, 681, 684, 696, 702, 705, 708, 716, 719, 726, 730, 739, 743, 
                    746, 758, 765, 766, 777, 783, 785, 786, 790, 792, 800, 802, 808, 809, 810, 816, 819, 823, 827, 835, 837, 842, 
                    845, 853, 865, 868, 873, 885, 889, 891, 897, 904, 906, 918, 941, 949, 956, 961, 969, 973, 976, 981, 984, 991, 
                    996, 999, 1002], 
                   [  0,  59,  68,  73,  76,  86,  94,  97, 101, 104, 109, 118, 120, 123, 128, 134, 136, 139, 143, 145, 148, 154, 
                    160, 162, 177, 190, 195, 199, 204, 208, 212, 220, 221, 222, 232, 234, 246, 251, 253, 256, 263, 270, 273, 287, 
                    302, 314, 322, 355, 357, 368, 396, 397, 401, 416, 428, 433, 437, 442, 446, 452, 456, 469, 475, 490, 500, 506, 
                    525, 526, 538, 545, 551, 554, 558, 562, 571, 579, 582, 586, 587, 600, 612, 664, 823, 824, 832, 850, 858, 870, 
                    879, 881, 889, 898, 901, 913, 916, 924, 932, 936, 943, 952, 955, 975, 985, 992, 997, 1001], 
                   [  0,  36,  56,  94, 114, 121, 136, 140, 143, 153, 159, 162, 171, 174, 182, 190, 197, 201, 215, 219, 237, 239, 
                    249, 251, 254, 265, 269, 297, 322, 335, 340, 343, 348, 352, 355, 363, 368, 371, 375, 384, 388, 395, 405, 411, 
                    439, 461, 464, 468, 477, 483, 485, 487, 494, 497, 501, 503, 514, 519, 521, 526, 529, 532, 538, 547, 551, 555, 
                    560, 579, 584, 597, 600, 602, 607, 609, 617, 623, 626, 634, 638, 644, 649, 652, 663, 671, 672, 675, 678, 685, 
                    688, 699, 700, 712, 719, 724, 726, 735, 744, 753, 756, 775, 797, 806, 812, 818, 826, 829, 832, 848, 852, 856, 
                    866, 869, 872, 875, 881, 886, 891, 893, 895, 901, 910, 930, 936, 940, 953, 958, 989, 997, 1004, 1006] 
                  ]
        return index
