import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
import pickle

# import pickle
# with open('x.pkl', 'wb') as file:
#     pickle.dump(object, file)

default_data_path = {
    "200MB": "/miniscratch/tyz/datasets/CloudCast/200MB/pkls/",
    "8GB": "/miniscratch/tyz/datasets/CloudCast/8GB/pkls/",
}


def load_cs_small(root=default_data_path):
    path_train = os.path.join(root["200MB"], "train.pkl")
    path_test = os.path.join(root["200MB"], "test.pkl")
    with open(path_train, "rb") as file:
        data_train = pickle.load(file)
        data_train = data_train
    with open(path_test, "rb") as file:
        data_test = pickle.load(file)
    return [data_train, data_test]


class CloudCast(data.Dataset):
    def __init__(
        self,
        root,
        is_train,
        n_frames_input,
        n_frames_output,
        is_large=False,
        max_pxl_value=20,
        transform=None,
        batchsize=16,
    ):
        """
        param num_objects: a list of number of possible objects.
        """
        super(CloudCast, self).__init__()

        self.dataset_all = load_cs_small()
        if is_train:
            self.dataset = self.dataset_all[0]
        else:
            self.dataset = self.dataset_all[1]
        self.length = self.dataset.shape[-1]
        self.is_large = False
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        self.max_pxl_value = max_pxl_value
        self.batchsize = batchsize
        # For generating data
        if self.is_large:
            self.image_size_ = 728
        else:
            self.image_size_ = 128
        self.step_length_ = 0.1

    def getslice(self, cloudcast, idx):
        # cloudcast is an ndarray with shape: (H,W,T)
        # idx is the index of the slice
        # this function aims to return the slice of the "video" with given idx
        # target shape: (n_frames_input + n_frames_output, H, W, C)
        H, W, T = cloudcast.shape
        num_normal_batch = int((self.length - 1) / self.batchsize)
        num_normal_data = num_normal_batch * self.batchsize

        if idx <= num_normal_data:
            slice = cloudcast[
                :, :, idx : idx + self.n_frames_total
            ]  # get a compelete slice from the begining
        else:  # avoid getting errors when the rest of the data is not enough for a batch
            diff = self.length - idx
            slice = cloudcast[:, :, -diff - self.n_frames_total : -diff]
        slice = slice.reshape((-1, 1, H, W))
        return slice

    def __getitem__(self, idx):
        #         if self.length < idx-1:
        #             return [idx, torch.empty(), torch.empty(), torch.empty(), np.zeros(1)]
        images = self.getslice(self.dataset, idx)
        input = images[: self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[
                -self.n_frames_output :
            ]  # avoid error when the rest of the data is not enough for an output with len of n_frames_output
        else:
            output = []
        frozen = input[-1]
        output = torch.from_numpy(output / self.max_pxl_value).contiguous().float()
        input = torch.from_numpy(input / self.max_pxl_value).contiguous().float()
        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return self.length
