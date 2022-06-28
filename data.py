import os
from random import randrange

import numpy as np
import h5py
import librosa
import pandas as pd
import torch
from utils import SPEAKERS_CONFIG

class S2G_Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.speaker = args.speaker
        self.base_path = args.base_path
        df = pd.read_csv(os.path.join(args.base_path, "train.csv"))
        df = df[df["speaker"] == args.speaker]
        self.df = df[df["dataset"] == "train"]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pose_fn = self.df.iloc[index]["pose_fn"]
        data = np.load(os.path.join(self.base_path, pose_fn))
        audio = data["audio"]

        mel = librosa.feature.melspectrogram(
            audio,
            sr=16000,
            hop_length=260,
            n_fft=400,
            fmin=125,
            fmax=7500,
            n_mels=64,
            center=False,
        )

        logmel = librosa.power_to_db(mel)
        mel_len = logmel.shape[1]
        logmel = logmel[:, : mel_len // 4 * 4]
        logmel = logmel.reshape(64 * 4, mel_len // 4)
        logmel = logmel.transpose()

        pose = data["pose"]
        if pose.shape[2] > 49:
            pose = np.delete(pose, [7, 8, 9], axis=2)

        pose -= pose[:, :, 0:1]
        pose = (pose - SPEAKERS_CONFIG[self.args.speaker]["mean"].reshape(2, 49)) / (
            SPEAKERS_CONFIG[self.args.speaker]["std"].reshape(2, 49)
            + np.finfo(float).eps
        )

        pose = pose[:, :, 1:].reshape(self.args.seq_len, self.args.joint_num * 2)

        return {"audios": logmel, "poses": pose}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.ids = []
        self.seq_len = args.seq_len
        # self.joint_num = args.joint_num

        self.file_names = os.listdir(args.base_path)
        self.file_names = [
            file_name for file_name in self.file_names if file_name.endswith(".h5")
        ]
        self.file_names = sorted(self.file_names)
        self.audio_features = []
        self.pose_features = []

        audio_mean = 0
        audio_var = 1
        if args.audio_stat and os.path.isfile(args.audio_stat):
            with open(args.audio_stat, "rb") as f:
                audio_mean = np.load(f)
                audio_var = np.load(f)
        hop_length = args.sr // args.fr
        for seq_id, file_name in enumerate(self.file_names):
            if not file_name.endswith(".h5"):
                continue
            with h5py.File(os.path.join(args.base_path, file_name), "r") as f:
                audio = f[args.audio_key][()]
                poses = f[args.pose_key][()]
                poses = poses[:, :, :3, :3]

            if poses.shape[0] >= args.seq_len:
                audio_feature = librosa.feature.melspectrogram(
                    y=audio, sr=args.sr, hop_length=hop_length, n_mels=64
                )
                audio_feature = librosa.power_to_db(audio_feature)
                audio_feature = audio_feature.transpose()
                audio_feature -= audio_mean
                audio_feature /= audio_var

                audio_len = audio_feature.shape[0]
                pose_len = poses.shape[0]
                seq_len = min(audio_len, pose_len)
                if args.seq_len > 0:
                    self.ids.extend([seq_id] * (seq_len // self.seq_len))
                else:
                    self.ids.extend([seq_id])
                audio_feature = audio_feature[:seq_len]
                poses = poses[:seq_len]

                self.audio_features.append(audio_feature)
                self.pose_features.append(poses)

    def __len__(self):
        return len(self.ids)  

    def __getitem__(self, index):
        audio_feature = self.audio_features[self.ids[index]]
        poses = self.pose_features[self.ids[index]]

        filename = self.file_names[self.ids[index]]
        if self.seq_len > 0:
            start = randrange(0, audio_feature.shape[0] - self.seq_len + 1)
            end = start + self.seq_len
        else:
            start = 0
            end = -1

        audio_feature = audio_feature[start:end]
        poses = poses[start:end]

        return {
            "audios": audio_feature,
            "poses": poses,
            "filename": filename,
        }
