import logging

import torch
from model import Model
from data import Dataset, S2G_Dataset
from options import options

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = options()
    # data = S2G_Dataset(args)
    data = Dataset(args)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    model = Model(args)
    if args.resume:
        model.resume(args.resume)
    model.train(dataloader)
