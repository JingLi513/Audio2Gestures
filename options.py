import os
import logging
import argparse
import torch
import multiprocessing


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    # model
    parser.add_argument("--dataset", type=str, required=True, help="Trinity/Speech2Gestures")
    parser.add_argument("--audio_size", type=int, default=64)
    parser.add_argument("--speaker",type=str, default="ellen")
    parser.add_argument("--joint_num", type=int, default=22)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--audio_hidden_size", type=int, default=16)
    parser.add_argument("--pose_hidden_size", type=int, default=16)
    parser.add_argument("--with_code_constrain", action="store_true")
    parser.add_argument("--with_mapping_net", action="store_true")
    parser.add_argument("--with_cyc", action="store_true")
    parser.add_argument("--with_ds", action="store_true")
    parser.add_argument("--with_translation", action="store_true")
    parser.add_argument("--with_audio_share_vae", action="store_true")
    parser.add_argument("--with_motion_share_vae", action="store_true")
    parser.add_argument("--with_motion_spec_vae", action="store_true")
    parser.add_argument("--with_mapping_net_vae", action="store_true")
    parser.add_argument("--using_mspec_stat",action="store_true")
    parser.add_argument("--using_2D_data",action="store_true")
    parser.add_argument("--save_freq",type=int,default=100)
    # data
    parser.add_argument("--base_path", type=str, default="data")
    parser.add_argument(
        "--smpl_path",
        type=str,
        default="SMPLX/SMPLX_FEMALE.pkl",
    )
    parser.add_argument("--audio_key", type=str, default="wave")
    parser.add_argument("--pose_key", type=str, default="joint_trans_mats")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--fr", type=int, default=30)
    # train
    parser.add_argument(
        "--optim", type=str, default="Adam", help="Adam/AdamW/RMSProp/SGD"
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--lambda_audio", type=float, default=1)
    parser.add_argument("--lambda_rotmat", type=float, default=1)
    parser.add_argument("--lambda_pose", type=float, default=1)
    parser.add_argument("--lambda_speed", type=float, default=1)
    parser.add_argument("--lambda_xrotmat", type=float, default=1)
    parser.add_argument("--lambda_xpose", type=float, default=1)
    parser.add_argument("--lambda_xspeed", type=float, default=1)

    parser.add_argument("--lambda_kl", type=float, default=1e-4)
    parser.add_argument("--lambda_code", type=float, default=0.1)
    parser.add_argument("--lambda_cyc", type=float, default=0.1)
    parser.add_argument("--lambda_ds", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--grad_norm", type=float, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--result_path", type=str, default="result")

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(args.name) > 0:
        args.ckpt_dir = args.ckpt_dir + "_" + args.name
        args.log_dir = args.log_dir + "_" + args.name
    logging.info(args)
    return args
