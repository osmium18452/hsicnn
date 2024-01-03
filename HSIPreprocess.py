import numpy as np
import torch
from scipy import io
from torch.utils.data import Dataset


class HSIPreprocess:
    def __init__(self, dataset, patch_size, ratio):
        if dataset == 'pc':
            self.data = io.loadmat('Datasets/PaviaC/Pavia.mat')['pavia']
            self.label = io.loadmat('Datasets/PaviaC/Pavia_gt.mat')["pavia_gt"]
        elif dataset == 'pu':
            self.data = io.loadmat('Datasets/PaviaU/PaviaU.mat')["paviaU"]
            self.label = io.loadmat('Datasets/PaviaU/PaviaU_gt.mat')["paviaU_gt"]
        elif dataset == 'sa':
            self.data = io.loadmat('Datasets/SalinasA/SalinasA_corrected.mat')["salinasA_corrected"]
            self.label = io.loadmat('Datasets/SalinasA/SalinasA_gt.mat')["salinasA_gt"]
        elif dataset == 'sv':
            self.data = io.loadmat('Datasets/Salinas/Salinas_corrected.mat')["salinas_corrected"]
            self.label = io.loadmat('Datasets/Salinas/Salinas_gt.mat')["salinas_gt"]
        else:
            print('Dataset Error')
            exit()
        self.patch_size = patch_size
        self.num_classes = len(np.unique(self.label)) - 1
        self.height = self.data.shape[0]
        self.width = self.data.shape[1]
        self.bands = self.data.shape[2]

        self.data = self.data.astype(float)
        for band in range(self.bands):
            self.data[:, :, band] = (self.data[:, :, band] - np.min(self.data[:, :, band])) / \
                                    (np.max(self.data[:, :, band]) - np.min(self.data[:, :, band]))
        pad_size = self.patch_size // 2
        self.data = np.pad(self.data, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), "symmetric")
        unique = np.unique(self.label)
        lut = np.zeros(np.max(unique) + 1, dtype=int)
        for iter, i in enumerate(unique):
            lut[i] = iter
        self.label = lut[self.label]

        self.class_patches = []
        self.class_index = []
        for i in range(self.num_classes):
            self.class_patches.append([])
            self.class_index.append([])
        for i in range(self.height):
            for j in range(self.width):
                tmp_label = self.label[i, j]
                tmp_patch = self.__patch(i, j)
                if tmp_label != 0:
                    self.class_patches[tmp_label - 1].append(tmp_patch)
                    self.class_index[tmp_label - 1].append(i * self.width + j)
        self.num_each_class = []
        for i in range(self.num_classes):
            self.num_each_class.append(len(self.class_patches[i]))

        self.train_set = []
        self.test_set = []
        self.train_labels = []
        self.test_labels = []
        for i in range(self.num_classes):
            label = i
            index = np.random.choice(self.num_each_class[label], int((self.num_each_class[label]) * ratio + 0.5),
                                     replace=False)
            self.train_set.extend(self.class_patches[label][j] for j in index)
            self.train_labels.extend(label for j in range(len(index)))

            index = np.setdiff1d(range(self.num_each_class[label]), index)
            self.test_set.extend(self.class_patches[label][j] for j in index)
            self.test_labels.extend(label for j in range(len(index)))

        self.train_set = torch.tensor(np.array(self.train_set),dtype=torch.float).transpose(1,-1)
        self.train_labels = self.__convert_to_onehot(torch.tensor(np.array(self.train_labels)))
        self.test_set = torch.tensor(np.array(self.test_set),dtype=torch.float).transpose(1,-1)
        self.test_labels = self.__convert_to_onehot(torch.tensor(np.array(self.test_labels)))


    def __patch(self, i, j):
        heightSlice = slice(i, i + self.patch_size)
        widthSlice = slice(j, j + self.patch_size)
        return self.data[heightSlice, widthSlice, :]

    def __convert_to_onehot(self, labels):
        onehot = torch.zeros(labels.shape[0], self.num_classes)
        for i in range(labels.shape[0]):
            onehot[i, labels[i]] = 1
        return onehot


class HSIDataset(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label

    def __getitem__(self, item):
        return self.input[item], self.label[item]

    def __len__(self):
        return self.input.shape[0]


if __name__ == '__main__':
    HSIPreprocess('pc', 5, .3)

