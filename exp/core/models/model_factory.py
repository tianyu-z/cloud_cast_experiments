import os
import torch
import torch.optim as optim
from core.models import predrnn, predrnn_memory_decoupling


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        # print(self.configs)
        self.num_hidden = [int(x) for x in configs.num_hidden.split(",")]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            "predrnn": predrnn.RNN,
            "predrnn_memory_decoupling": predrnn_memory_decoupling.RNN,
        }
        print(self.num_layers, self.num_hidden)
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(
                configs.device
            )
        else:
            raise ValueError("Name of network unknown %s" % configs.model_name)

        self.optimizer = optim.Adam(self.network.parameters(), lr=configs.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=configs.milestones, gamma=0.1
        )

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def save(self, itr):
        stats = {}
        stats["lr"] = self.get_lr(self.optimizer)
        stats["net_param"] = self.network.state_dict()
        checkpoint_path = os.path.join(
            self.configs.save_dir, "model.ckpt" + "-" + str(itr)
        )
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def save_best(self, metric):
        stats = {}
        stats["lr"] = self.get_lr(self.optimizer)
        stats["net_param"] = self.network.state_dict()
        checkpoint_path = os.path.join(
            self.configs.save_dir, "model.ckpt" + "-best" + metric
        )
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print("load model:", checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats["net_param"])
        self.optimizer = optim.Adam(self.network.parameters(), lr=stats["lr"])

    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames.cpu()).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        _, loss = self.network(frames_tensor, mask_tensor)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames.cpu()).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
