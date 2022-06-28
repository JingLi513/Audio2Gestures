import os
import numpy as np
import h5py
import torch

from model import Model
from data import Dataset
from options import options


def calculate_training_data_stats(model, dataloader, args):
    """
    Extracting motion specific code stats from training data
    """
    means = []
    vars = []
    for data in dataloader:
        clip_len = 64
        start = 0
        end = start + clip_len
        poses = data["poses"].to(args.device)
        poses = model.motion_processor.encode_motion(poses)
        seq_len = poses.shape[1]
        while end < seq_len:

            _, z_motion_spec = model.net_G["motion_enc"](poses[:, start:end].float())
            
            mean = z_motion_spec.mean(dim=(1,)).cpu().detach().numpy()
            var = z_motion_spec.std(dim=(1,)).cpu().detach().numpy()
            assert not np.any(np.isnan(mean))
            assert not np.any(np.isnan(var))
            means.append(mean)
            vars.append(var)
            start = end
            end = start + clip_len
    means = np.concatenate(means, axis=0)
    vars = np.concatenate(vars, axis=0)
    if not os.path.exists(os.path.join(args.result_path, "stat")):
        os.makedirs(os.path.join(args.result_path, "stat"))
    with h5py.File(f"{args.result_path}/stat/motion_spec_stat.h5", "w") as f:
        f.create_dataset("means", data=means)
        f.create_dataset("vars", data=vars)


if __name__ == "__main__":
    args = options()
    data = Dataset(args)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    model = Model(args)
    model.resume(args.resume)
    model.net_G.eval()
    calculate_training_data_stats(model, dataloader, args)
