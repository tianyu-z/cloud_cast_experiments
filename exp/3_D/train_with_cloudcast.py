from data.cloudcast import CloudCast
import torch
from tqdm import tqdm

trainFolder = CloudCast(
    is_train=True, root="data/", n_frames_input=10, n_frames_output=1, batchsize=16,
)
validFolder = CloudCast(
    is_train=False, root="data/", n_frames_input=10, n_frames_output=1, batchsize=16,
)
trainLoader = torch.utils.data.DataLoader(
    trainFolder, batch_size=16, num_workers=0, shuffle=False
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for epoch in range(1):
    # Training

    t = tqdm(trainLoader, leave=False, total=len(trainLoader))
    for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
        inputs = inputVar.to(device)

        break
