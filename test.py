import os
import h5py
import torch

from model import Model
from data import Dataset
from options import options



def test_audio2pose(model, dataloader, args):
    n_repeat = 10

    for i in range(n_repeat):
        for data in dataloader:
            audios = data["audios"].float().to(args.device)
            pred_motion = model.inference(audios)
            rot_mats = model.motion_processor.decode_motion(pred_motion)
            poses = model.motion_processor.calculate_pos(rot_mats)
            result_dir = f"{args.result_path}/{args.name}/{i}"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            with h5py.File(f"{result_dir}/{data['filename'][0]}", "w") as f:
                print("writing", data["filename"][0])
                f.create_dataset(
                    "LclRotation", data=rot_mats.squeeze(0).cpu().detach().numpy()
                )
                f.create_dataset("poses", data=poses.squeeze(0).cpu().detach().numpy())
                f.create_dataset("wave", data=data["audios"].squeeze(0))




if __name__ == "__main__":
    args = options()
    data = Dataset(args)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    model = Model(args)
    model.resume(args.resume)
    model.net_G.eval()
    test_audio2pose(model, dataloader, args)
