import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class dataset():

    def __init__(self, match_list, start_frame, end_frame):
        self.datapath = r"../1-generate/data/"
        self.available_targets = self.get_actions_list(zigzag=False)
        self.match_list = match_list
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.data, self.targets = self.setDataFromMatchAndNormalize()
        self.data_size = self.data.shape
        self.target_size = self.targets.shape
        self.device = self.set_device()

    def setDataFromMatchAndNormalize(self):
        frames_arr = []
        actions_arr = []

        for m in self.match_list:
            
            _, frames, actions, _, _ = self.load_npz(self.datapath, m)
            frames = frames[self.start_frame - 1:self.end_frame, 30:130, 10:110]
            actions = actions[self.start_frame - 1:self.end_frame]
            
            action_one_hot = [self.prepare_action_data(i, self.available_targets) for i in actions]
            actions = np.array(action_one_hot)
            actions = actions.reshape(len(actions), -1)
            
            frames_arr.append(frames)
            actions_arr.append(actions)

        frames_arr = np.array(frames_arr)/255
        actions_arr = np.array(actions_arr)

        return frames_arr, actions_arr

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

    def prepare_action_data(self, action, ACTIONS_LIST):

        new_action = np.zeros((1, len(ACTIONS_LIST)), dtype=int) 

        new_action[0, ACTIONS_LIST.index(action)] = 1

        return new_action

    def set_device(self):

        use_gpu = self.ask_gpu()
    
        if use_gpu == 'y' and torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("Selected CPU")
            
        return device

    def ask_gpu(self):
        return input(("Do you want to use GPU (y/n)"))

class chunk_dataset(dataset):
    def __init__(self, match_list, start_frame, end_frame):
        super().__init__(match_list, start_frame, end_frame)
        self.chunk_idx = self.chunk_idx()
        self.data, self.targets, self.seq_len = self.setChunkedData(self.data, self.targets)

    def setChunkedData(self, data, targets):
        
        data, targets = self.chunkSequence(data, targets)
        data, targets = self.toNumpyArray(data, targets)
        data, targets = self.filteringSeqMoreThan(data, targets, 100)
        data, targets = self.tolistOfTensor(data, targets)
        seq_len = self.setSeqLen(data)

        return data, targets, seq_len

    def chunkSequence(self, data, targets):
        data_chunked = []
        target_chunked = []
        for i in range(len(data)):
            for j in range(len(self.chunk_idx[i]) - 1):
                data_chunked.append(data[i][self.chunk_idx[i][j]:self.chunk_idx[i][j+1]])
                target_chunked.append(targets[i][self.chunk_idx[i][j]:self.chunk_idx[i][j+1]])
        return data_chunked, target_chunked

    def toNumpyArray(self, X_train, Y_train):
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        return X_train, Y_train

    def filteringSeqMoreThan(self, X_train, Y_train, max_len):
        X_train_filtrado = []
        Y_train_filtrado = []
        for i in range(len(X_train)):
            if len(X_train[i]) > max_len:
                pass
            else:
                X_train_filtrado.append(X_train[i])
                Y_train_filtrado.append(Y_train[i])
        return X_train_filtrado, Y_train_filtrado
    
    def tolistOfTensor(self, X_train, Y_train):
        X_train = list(map(lambda x: torch.tensor(x), X_train))
        Y_train = list(map(lambda x: torch.tensor(x), Y_train))
        return X_train, Y_train
    
    def setSeqLen(self, X_train):
        return torch.FloatTensor(list(map(len,X_train)))
    
    def padSequence(self):
        self.data = pad_sequence(self.data, batch_first=True).float()
        self.targets = pad_sequence(self.targets, batch_first=True).float()

    def packPaddedSequence(self):
        self.data = pack_padded_sequence(self.data, self.seq_len, batch_first=True, enforce_sorted=False)

    def chunk_idx(self):
        indices = [ # match 1
                   [  0,  62,  80, 106, 117, 137, 142, 156, 159, 166, 185, 215, 226, 235, 242, 252, 259, 279, 281, 292, 296, 299, 
                    303, 308, 315, 318, 321, 329, 333, 335, 354, 360, 374, 377, 382, 386, 390, 393, 402, 414, 420, 444, 447, 465, 
                    469, 487, 491, 504, 506, 510, 513, 525, 528, 538, 552, 558, 570, 576, 581, 584, 588, 595, 598, 604, 607, 610, 
                    618, 619, 622, 631, 641, 645, 662, 672, 685, 693, 697, 704, 718, 724, 726, 735, 739, 745, 754, 755, 770, 773, 
                    789, 790, 799, 805, 809, 823, 834, 850, 854, 857, 868, 874, 875, 877, 883, 885, 904, 906, 911, 919, 923, 926, 
                    932, 943, 962, 965, 984, 991, 999, 1007, 1010, 1015, 1017, 1018],
                   # match 2
                   [  0,  31,  45,  47,  56,  63,  69,  85,  88,  92,  96, 112, 118, 124, 130, 137, 143, 151, 158, 170, 184, 202, 
                    223, 243, 249, 255, 260, 269, 272, 275, 276, 282, 285, 295, 296, 297, 306, 308, 318, 322, 331, 334, 337, 338, 
                    341, 343, 354, 357, 381, 387, 400, 402, 408, 420, 423, 426, 429, 441, 443, 445, 448, 454, 463, 465, 474, 481,
                    502, 506, 524, 533, 538, 548, 556, 562, 566, 570, 581, 588, 603, 604, 609, 652, 668, 681, 698, 710, 714, 725, 
                    729, 736, 739, 743, 748, 749, 752, 766, 770, 774, 778, 781, 792, 795, 803, 804, 806, 814, 818, 821, 828, 834, 
                    837, 843, 849, 855, 862, 868, 874, 884, 888, 895, 896, 919, 935, 942, 945, 957, 958, 963, 964, 965, 977, 980,
                    984, 989, 997, 1000], 
                   # match 3
                   [  0,  37,  53,  81,  94,  99, 119, 123, 124, 142, 144, 146, 162, 193, 211, 213, 222, 230, 232, 235, 238, 240, 
                    252, 255, 258, 267, 270, 275, 278, 284, 287, 289, 300, 307, 313, 331, 352, 359, 371, 379, 386, 389, 392, 409, 
                    412, 432, 435, 440, 442, 446, 462, 467, 472, 506, 527, 531, 535, 553, 557, 561, 571, 579, 589, 602, 607, 609, 
                    612, 616, 621, 631, 634, 638, 640, 644, 645, 661, 669, 673, 676, 692, 695, 698, 701, 711, 717, 720, 721, 734, 
                    744, 747, 750, 755, 759, 774, 785, 793, 807, 859, 891, 895, 912, 927, 948, 958, 969, 980, 997, 998, 1004], 
                   # match 4
                   [  0,  33,  58,  75,  86, 100, 118, 120, 139, 142, 147, 159, 162, 165, 167, 174, 179, 180, 183, 192, 197, 199, 
                    209, 214, 217, 222, 227, 231, 232, 239, 244, 251, 262, 266, 271, 275, 277, 282, 287, 291, 294, 297, 308, 311, 
                    316, 318, 322, 327, 339, 342, 348, 352, 362, 372, 383, 397, 399, 402, 405, 417, 420, 425, 428, 454, 457, 464, 
                    470, 477, 481, 485, 488, 494, 495, 503, 507, 517, 535, 537, 540, 553, 556, 558, 577, 588, 598, 601, 609, 620, 
                    634, 638, 647, 650, 656, 659, 660, 663, 669, 672, 681, 684, 696, 702, 705, 708, 716, 719, 726, 730, 739, 743, 
                    746, 758, 765, 766, 777, 783, 785, 786, 790, 792, 800, 802, 808, 809, 810, 816, 819, 823, 827, 835, 837, 842, 
                    845, 853, 865, 868, 873, 885, 889, 891, 897, 904, 906, 918, 941, 949, 956, 961, 969, 973, 976, 981, 984, 991, 
                    996, 999, 1002], 
                   # match 5
                   [  0,  59,  68,  73,  76,  86,  94,  97, 101, 104, 109, 118, 120, 123, 128, 134, 136, 139, 143, 145, 148, 154, 
                    160, 162, 177, 190, 195, 199, 204, 208, 212, 220, 221, 222, 232, 234, 246, 251, 253, 256, 263, 270, 273, 287, 
                    302, 314, 322, 355, 357, 368, 396, 397, 401, 416, 428, 433, 437, 442, 446, 452, 456, 469, 475, 490, 500, 506, 
                    525, 526, 538, 545, 551, 554, 558, 562, 571, 579, 582, 586, 587, 600, 612, 664, 823, 824, 832, 850, 858, 870, 
                    879, 881, 889, 898, 901, 913, 916, 924, 932, 936, 943, 952, 955, 975, 985, 992, 997, 1001], 
                   # match 6
                   [  0,  36,  56,  94, 114, 121, 136, 140, 143, 153, 159, 162, 171, 174, 182, 190, 197, 201, 215, 219, 237, 239, 
                    249, 251, 254, 265, 269, 297, 322, 335, 340, 343, 348, 352, 355, 363, 368, 371, 375, 384, 388, 395, 405, 411, 
                    439, 461, 464, 468, 477, 483, 485, 487, 494, 497, 501, 503, 514, 519, 521, 526, 529, 532, 538, 547, 551, 555, 
                    560, 579, 584, 597, 600, 602, 607, 609, 617, 623, 626, 634, 638, 644, 649, 652, 663, 671, 672, 675, 678, 685, 
                    688, 699, 700, 712, 719, 724, 726, 735, 744, 753, 756, 775, 797, 806, 812, 818, 826, 829, 832, 848, 852, 856, 
                    866, 869, 872, 875, 881, 886, 891, 893, 895, 901, 910, 930, 936, 940, 953, 958, 989, 997, 1004, 1006],
                   # match 7
                   [ 13,  34,  56,  77,  83,  98, 103, 109, 114, 119, 121, 126, 129, 131, 136, 137, 138, 144, 147, 154, 157, 159,
                    172, 174, 178, 191, 193, 195, 208, 210, 221, 224, 226, 232, 240, 243, 251, 257, 266, 285, 299, 303, 307, 319, 
                    324, 328, 330, 334, 342, 345, 348, 352, 354, 361, 364, 369, 371, 374, 379, 382, 387, 393, 397, 400, 406, 411, 
                    418, 423, 428, 433, 436, 440, 445, 450, 459, 465, 468, 470, 489, 495, 497, 507, 510, 546, 557, 584, 589, 592, 
                    608, 620, 624, 634, 649, 659, 661, 668, 670, 683, 690, 693, 696, 705, 723, 731, 748, 754, 770, 776, 779, 784, 
                    786, 794, 797, 802, 807, 808, 819, 823, 825, 827, 834, 843, 856, 863, 865, 872, 880, 882, 883, 888, 891, 894, 
                    899, 902, 904, 906, 912, 920, 923, 931, 939, 948, 956, 970, 973, 981, 989, 992, 997, 999, 1008, 1010, 1013, 1014],
                   # match 8
                   [ 37,  57,  73,  83, 100, 119, 130, 141, 147, 155, 159, 161, 172, 174, 186, 188, 193, 195, 201, 203, 207, 209, 
                    218, 220, 223, 225, 232, 236, 239, 241, 243, 244, 247, 259, 261, 263, 279, 294, 300, 302, 305, 307, 317, 319, 
                    321, 324, 327, 331, 336, 337, 338, 347, 350, 351, 359, 363, 365, 374, 379, 386, 411, 422, 437, 446, 466, 469, 
                    473, 492, 498, 510, 513, 524, 531, 534, 538, 550, 557, 572, 575, 594, 601, 608, 616, 619, 626, 634, 638, 641, 
                    642, 650, 653, 661, 665, 668, 676, 687, 714, 725, 736, 746, 747, 760, 770, 784, 794, 807, 817, 831, 842, 846, 
                    854, 864, 867, 871, 874, 878, 881, 894, 898, 901, 904, 914, 925, 928, 934, 937, 938, 947, 951, 960, 967, 970, 
                    974, 979, 987, 990, 993, 996, 999, 1005, 1013, 1016, 1019, 1025],
                   # match 9
                   [ 13,  36,  58,  77,  85, 103, 108, 114, 119, 123, 125, 130, 131, 137, 142, 150, 156, 160, 166, 173, 176, 179, 
                    182, 183, 188, 192, 194, 199, 202, 204, 206, 218, 221, 222, 226, 228, 231, 243, 245, 246, 248, 252, 253, 262, 
                    269, 275, 279, 281, 284, 291, 298, 300, 302, 306, 313, 315, 318, 328, 334, 336, 357, 360, 371, 376, 377, 379, 
                    388, 391, 398, 402, 410, 420, 423, 431, 440, 441, 444, 450, 453, 460, 462, 471, 474, 482, 494, 507, 518, 521, 
                    528, 535, 540, 542, 552, 555, 567, 570, 573, 575, 585, 588, 591, 595, 604, 607, 613, 618, 623, 628, 637, 638, 
                    647, 655, 660, 669, 678, 687, 700, 717, 721, 729, 744, 747, 748, 760, 772, 775, 788, 789, 800, 818, 819, 824, 
                    848, 851, 870, 878, 887, 896, 897, 899, 900, 903, 907, 925, 927, 932, 937, 940, 955, 962, 965, 968, 979, 990, 
                    992, 1000, 1010, 1018],
                   # match 10
                   [ 14,  36,  58,  77,  85, 103, 108, 114, 119, 123, 125, 130, 131, 137, 142, 150, 156, 160, 166, 173, 176, 179, 
                    182, 183, 188, 192, 194, 199, 202, 204, 206, 218, 221, 222, 226, 228, 231, 243, 245, 246, 248, 252, 253, 262, 
                    269, 275, 279, 281, 284, 291, 298, 300, 302, 306, 313, 315, 318, 328, 334, 336, 357, 360, 371, 376, 377, 379, 
                    388, 391, 398, 402, 410, 420, 423, 431, 440, 441, 444, 450, 453, 460, 462, 471, 474, 482, 494, 507, 518, 521, 
                    528, 535, 540, 542, 552, 554, 566, 570, 573, 575, 585, 588, 591, 595, 604, 607, 613, 618, 623, 628, 637, 638, 
                    647, 655, 660, 669, 678, 687, 700, 717, 721, 729, 744, 747, 748, 760, 772, 775, 788, 789, 800, 818, 819, 824, 
                    848, 851, 870, 878, 897, 899, 900, 903, 907, 925, 927, 932, 936, 940, 955, 962, 965, 968, 979, 990, 992, 1000, 
                    1010, 1018]
                  ]
        return indices