import os
import logging
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils import tensorboard
from torch.distributions import Normal

from module import ConvNet
from utils import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    load_smplx_model,
    blend_shapes,
    vertices2joints,
    batch_rigid_transform,
    so3_relative_angle,
    SPEAKERS_CONFIG,
)


def init(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class Motion_Process(nn.Module):
    def __init__(self) -> None:
        super(Motion_Process, self).__init__()

    def encode_motion(self, motion: torch.Tensor):
        return motion

    def decode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return motion

    def calculate_pos(self, motion: torch.Tensor) -> torch.Tensor:
        return motion

    def calculate_joint_speed(self, pos: torch.Tensor) -> torch.Tensor:
        return pos[:, 1:] - pos[:, :-1]


class Process_3D_Motion(nn.Module):
    def __init__(
        self,
        smplx_path,
        ignore_joints=None, # set the rotation of the joints to identity
    ) -> None:
        super(Process_3D_Motion, self).__init__()
        if ignore_joints is None:
            self.ignore_joints = []
        else:
            self.ignore_joints = ignore_joints

        data_struct = load_smplx_model(smplx_path)
        self.register_buffer("parents", data_struct["parents"])
        self.joint_num = len(self.parents)
        meta_betas = torch.zeros(1, 20).float()
        v_shaped = data_struct["v_template"] + blend_shapes(meta_betas, data_struct["shapedirs"])
        self.register_buffer("J", vertices2joints(data_struct["J_regressor"], v_shaped))
        self.selected_joints = [i for i in range(self.joint_num) if i not in self.ignore_joints]

    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        assert len(motion.shape) == 5, f"Expects an array of size BxTxNx3x3, but received {motion.shape}"
        B, T = motion.shape[:2]
        # motion[:, :, self.ignore_joints] = torch.eye(3, device=motion.device)
        motion = motion[:, :, self.selected_joints, :2, :3].reshape(B, T, -1)
        return motion

    def decode_motion(self, motion: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        assert len(motion.shape) == 3, f"Expects an array of size BxTxC, but received {motion.shape}"
        B, T = motion.shape[:2]
        rot_mats = rotation_6d_to_matrix(motion.reshape(-1, 6)).reshape(B, T, -1, 3, 3)
        output = torch.eye(3, device=rot_mats.device).tile(B, T, self.joint_num, 1, 1)
        output[:, :, self.selected_joints] = rot_mats
        # rot_mats = rot_mats.reshape(B, T, -1, 3, 3)
        return output

    def calculate_pos(self, motion: torch.Tensor) -> torch.Tensor:
        assert len(motion.shape) == 5, f"Expects an array of size BxTxNx3x3, but received {motion.shape}"
        B, T = motion.shape[:2]
        motion[:, :, self.ignore_joints] = torch.eye(3, device=motion.device)
        rot_mats = motion.reshape(B * T, -1, 3, 3)
        poses, _ = batch_rigid_transform(
            rot_mats,
            self.J.expand(rot_mats.shape[0], self.joint_num, 3),
            self.parents,
            dtype=torch.float32,
        )
        poses = poses.reshape(B, T, -1, 3)
        return poses

    def calculate_joint_speed(self, pos: torch.Tensor) -> torch.Tensor:
        # assert len(pos.shape) == 4, f"Expects an array of size BxTxNxC, but received {pos.shape}"
        return pos[:, 1:] - pos[:, :-1]



class Process_S2G_Motion(Motion_Process):
    def __init__(self, args) -> None:
        super(Process_S2G_Motion, self).__init__()
        joint_ids = np.delete(np.arange(52), [7, 8, 9])
        self.register_buffer("joint_ids", torch.LongTensor(joint_ids))
        self.register_buffer(
            "mean",
            torch.FloatTensor(SPEAKERS_CONFIG[args.speaker]["mean"].reshape(2, 49)),
        )
        self.register_buffer(
            "std",
            torch.FloatTensor(
                SPEAKERS_CONFIG[args.speaker]["std"].reshape(2, 49)
                + np.finfo(float).eps
            ),
        )

    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        assert len(motion.shape) == 4 and motion.shape[2] == 52
        B, T = motion.shape[:2]
        motion = motion.take_along_dim(self.joint_ids, dim=2)
        motion -= motion[:, :, :, 0:1]
        motion = (motion - self.mean) / self.std
        motion = motion[:, :, 1:].reshape(B, T, 48 * 2)
        return motion

    def decode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        assert (
            len(motion.shape) == 3
        ), f"Expects an array of size BxTxC, but received {motion.shape}"
        B, T = motion.shape[:2]
        motion = motion.reshape(B, T, 2, 48)
        motion = torch.cat((torch.zeros(motion.shape[0], 2, 1), motion), dim=2)
        motion = motion * self.std + self.mean
        return motion


class Model:
    def __init__(self, args, mode="Train"):
        super().__init__()
        self.args = args
        if args.dataset == "Speech2Gestures":
            self.motion_processor = Process_S2G_Motion(args)
        elif args.dataset == "Trinity":
            self.motion_processor = Process_3D_Motion(args.smplx_path, ignore_joints=list(range(22,55)))
        else:
            raise NotImplementedError

        self.device = torch.device(args.device)
        self.logger = tensorboard.SummaryWriter(args.log_dir)
        self.net_G = torch.nn.ModuleDict(
            {
                "audio_enc": Audio_Enc(args),
                "motion_enc": Motion_Enc(args),
                "motion_dec": Motion_Dec(args),
                "mapping_net": MappingNet(args),
            }
        ).to(self.device)
        self.motion_processor.to(self.device)
        self.net_G.apply(init)

        if mode == "Train":
            self.optimG = self.init_optim(self.net_G.parameters())
            self.global_step = 0
            self.epoch = 0

    def log(self, batch, loss_dict):
        for key in loss_dict:
            self.logger.add_scalar(key, loss_dict[key].item(), self.global_step)
        if batch % self.args.log_freq == 0:
            logging.info(
                f"Name: {self.args.name}, Epoch: {self.epoch}, Batch: {batch}/{self.batch_counts_per_epoch}"
            )
            for key in loss_dict:
                logging.info(f"{key}: {loss_dict[key].item()}")

    def sampling(self, size=None, mean=None, var=None):
        if self.args.using_mspec_stat:
            normal = Normal(mean, var)
            z_x = normal.sample((size,)).permute(1, 0, 2)
        else:
            z_x = torch.randn(size, device=self.device)
        if self.args.with_mapping_net:
            z_x = self.net_G["mapping_net"](z_x)
        return z_x

    def inference(self, audios: torch.Tensor, motions=None):
        if self.args.seq_len > 0:
            seq_len = min(audios.shape[1], self.args.seq_len)
        else:
            seq_len = audios.shape[1]

        if motions is None:
            if self.args.using_mspec_stat:
                with h5py.File(
                    f"{self.args.result_path}/stat/motion_spec_stat.h5", "r"
                ) as f:
                    means = torch.FloatTensor(f["means"][()]).to(self.device)
                    vars = torch.FloatTensor(f["vars"][()]).to(self.device)

        z_audio_share = self.net_G["audio_enc"](audios[:, :seq_len])
        if motions is None:
            if self.args.using_mspec_stat:
                idx = random.randint(0, means.shape[0] - 1)
                z_motion_spec = self.sampling(
                    mean=means[idx : idx + 1], var=vars[idx : idx + 1], size=z_audio_share.shape[1]
                )
            else:
                z_motion_spec = self.sampling(size=z_audio_share.shape)
        else:
            _, z_motion_spec = self.net_G["motion_enc"](motions[:, :seq_len])
        pred_motions = self.net_G["motion_dec"](z_audio_share, z_motion_spec)
        return pred_motions

    def train_one_batch(self, audios: torch.Tensor, motions: torch.Tensor):
        self.z_audio_share = self.net_G["audio_enc"](audios)
        (self.z_motion_share, self.z_motion_specific,) = self.net_G["motion_enc"](
            motions
        )

        recon_m = self.net_G["motion_dec"](self.z_motion_share, self.z_motion_specific)
        a2m = self.net_G["motion_dec"](self.z_audio_share, self.z_motion_specific)
        self.z_x = self.sampling(
            size=self.z_motion_specific.shape[1],
            mean=self.z_motion_specific.mean(dim=(1,)),
            var=self.z_motion_specific.std(dim=(1,)),
        )

        a2x = self.net_G["motion_dec"](self.z_audio_share, self.z_x)

        (self.z_a2x_share, self.z_a2x_spec) = self.net_G["motion_enc"](
            self.motion_processor.encode_motion(self.motion_processor.decode_motion(a2x))
        )
        return recon_m, a2m, a2x

    def calculate_2d_loss(self, tgt_p, recon_p, a2m_p, a2x_p, batch):
        tgt_p = self.motion_processor.decode_motion(tgt_p)
        recon_p = self.motion_processor.decode_motion(recon_p)
        a2m_p = self.motion_processor.decode_motion(a2m_p)
        a2x_p = self.motion_processor.decode_motion(a2x_p)

        tgt_s = self.motion_processor.calculate_joint_speed(tgt_p)
        recon_s = self.motion_processor.calculate_joint_speed(recon_p)
        a2m_s = self.motion_processor.calculate_joint_speed(a2m_p)
        a2x_s = self.motion_processor.calculate_joint_speed(a2x_p)
        joint_distance = torch.abs(a2x_p - tgt_p)
        loss_G_dict = {
            "pos/recon_position": F.l1_loss(recon_p, tgt_p) * self.args.lambda_pose,
            "speed/recon_speed": F.l1_loss(recon_s, tgt_s) * self.args.lambda_speed,
            "pos/audio2position": F.l1_loss(a2m_p, tgt_p) * self.args.lambda_pose,
            "speed/audio2speed": F.l1_loss(a2m_s, tgt_s) * self.args.lambda_speed,
            "pos/audio2position_x": joint_distance[
                joint_distance > self.args.tolerance
            ].mean()
            * self.args.lambda_xpose,
            "speed/audio2speed_x": F.l1_loss(a2x_s, tgt_s) * self.args.lambda_xspeed,
        }
        if self.args.with_code_constrain:
            loss_G_dict.update(
                {
                    "code/share_code_constrain": F.l1_loss(
                        self.z_audio_share, self.z_motion_share
                    )
                    * self.args.lambda_code,
                }
            )
        if self.args.with_cyc:
            loss_G_dict.update(
                {
                    "code/cyc_spec": F.l1_loss(self.z_a2x_spec, self.z_x)
                    * self.args.lambda_cyc,
                    "code/cyc_share": F.l1_loss(self.z_a2x_share, self.z_motion_share) * self.args.lambda_cyc,
                }
            )
        if self.args.with_ds:
            loss_G_dict.update(
                {
                    "pos/diverse": -F.l1_loss(a2x_p, a2m_p.detach())
                    * self.args.lambda_ds,
                }
            )
        loss_G_dict.update(self.net_G["audio_enc"].get_loss_dict())
        loss_G_dict.update(self.net_G["motion_enc"].get_loss_dict())
        loss_G_dict.update(self.net_G["mapping_net"].get_loss_dict())
        self.log(batch, loss_G_dict)
        loss_G = torch.stack(list(loss_G_dict.values())).sum()
        return loss_G

    def calculate_3d_loss(self, tgt_m, recon_m, a2m, a2x, batch):
        tgt_r = self.motion_processor.decode_motion(tgt_m)
        recon_r = self.motion_processor.decode_motion(recon_m)
        a2m_r = self.motion_processor.decode_motion(a2m)
        a2x_r = self.motion_processor.decode_motion(a2x)

        tgt_p = self.motion_processor.calculate_pos(tgt_r)
        recon_p = self.motion_processor.calculate_pos(recon_r)
        a2m_p = self.motion_processor.calculate_pos(a2m_r)
        a2x_p = self.motion_processor.calculate_pos(a2x_r)

        tgt_s = self.motion_processor.calculate_joint_speed(tgt_p)
        recon_s = self.motion_processor.calculate_joint_speed(recon_p)
        a2m_s = self.motion_processor.calculate_joint_speed(a2m_p)
        a2x_s = self.motion_processor.calculate_joint_speed(a2x_p)

        joint_distance = torch.abs(a2x_p - tgt_p)

        loss_G_dict = {
            "rot/recon_rotmats": so3_relative_angle(recon_r, tgt_r).mean()
            * self.args.lambda_rotmat,
            "pos/recon_position": F.l1_loss(recon_p, tgt_p) * self.args.lambda_pose,
            "speed/recon_speed": F.l1_loss(recon_s, tgt_s) * self.args.lambda_speed,
            "rot/audio2motion": so3_relative_angle(a2m_r, tgt_r).mean()
            * self.args.lambda_rotmat,
            "pos/audio2position": F.l1_loss(a2m_p, tgt_p) * self.args.lambda_pose,
            "speed/audio2speed": F.l1_loss(a2m_s, tgt_s) * self.args.lambda_speed,
            "rot/audio2motion_x": so3_relative_angle(a2x_r, tgt_r).mean()
            * self.args.lambda_xrotmat,
            "pos/audio2position_x": joint_distance[
                joint_distance > self.args.tolerance
            ].mean()
            * self.args.lambda_xpose,
            "speed/audio2speed_x": F.l1_loss(a2x_s, tgt_s) * self.args.lambda_xspeed,
        }
        if self.args.with_code_constrain:
            loss_G_dict.update(
                {
                    "code/share_code_constrain": F.l1_loss(
                        self.z_audio_share, self.z_motion_share
                    )
                    * self.args.lambda_code,
                }
            )
        if self.args.with_cyc:
            loss_G_dict.update(
                {
                    "code/cyc_spec": F.l1_loss(self.z_a2x_spec, self.z_x)
                    * self.args.lambda_cyc,
                    "code/cyc_share": F.l1_loss(self.z_a2x_share, self.z_motion_share) * self.args.lambda_cyc,
                }
            )
        if self.args.with_ds:
            loss_G_dict.update(
                {
                    "pos/diverse": -F.l1_loss(a2x_p, a2m_p.detach())
                    * self.args.lambda_ds,
                }
            )
        loss_G_dict.update(self.net_G["audio_enc"].get_loss_dict())
        loss_G_dict.update(self.net_G["motion_enc"].get_loss_dict())
        loss_G_dict.update(self.net_G["mapping_net"].get_loss_dict())
        self.log(batch, loss_G_dict)
        loss_G = torch.stack(list(loss_G_dict.values())).sum()
        return loss_G

    def train(self, dataloader):
        os.makedirs(self.args.ckpt_dir, exist_ok=False)
        self.net_G.train()
        self.batch_counts_per_epoch = len(dataloader)

        init_lambda_ds = self.args.lambda_ds
        while self.epoch < self.args.epochs:
            data_iter = iter(dataloader)
            batch = 0
            while batch < self.batch_counts_per_epoch:
                data = data_iter.next()
                if self.args.lambda_ds > 0:
                    self.args.lambda_ds -= init_lambda_ds / 500000
                audios = data["audios"].float().to(self.device)
                motion = data["poses"].float().to(self.device)
                self.optimG.zero_grad()

                motion = self.motion_processor.encode_motion(motion)
                recon_m, a2m, a2x = self.train_one_batch(
                    audios, motion
                )
                if self.args.dataset in ["Trinity"]:
                    loss_G = self.calculate_3d_loss(
                        motion, recon_m, a2m, a2x, batch
                    )
                elif self.args.dataset in ["Speech2Gesture"]:
                    loss_G = self.calculate_2d_loss(
                        motion, recon_m, a2m, a2x, batch
                    )
                loss_G.backward()
                self.optimG.step()

                batch += 1
                self.global_step += 1

            self.epoch += 1
            if self.epoch % self.args.save_freq == 0:
                self.save(loss_G.item())

    def init_optim(self, param):
        if self.args.optim == "Adam":
            logging.info("Using Adam optimizer")
            logging.info(f"lr: {self.args.lr}")
            return torch.optim.Adam(param, lr=self.args.lr)
        elif self.args.optim == "AdamW":
            logging.info("Using AdamW optimizer")
            logging.info(f"lr: {self.args.lr}")
            return torch.optim.AdamW(param, lr=self.args.lr)
        elif self.args.optim == "RMSProp":
            logging.info("Using RMSProp optimizer")
            logging.info(f"lr: {self.args.lr}")
            return torch.optim.RMSprop(param, lr=self.args.lr)
        elif self.args.optim == "SGD":
            logging.info("Using SGD optimizer")
            logging.info(f"lr: {self.args.lr}, momentum: {self.args.momentum}")
            return torch.optim.SGD(param, lr=self.args.lr, momentum=self.args.momentum,)
        else:
            raise NotImplementedError

    def save(self, loss):
        state = {"args": self.args}
        state["net_G"] = self.net_G.state_dict()
        state["epoch"] = self.epoch
        state["global_step"] = self.global_step
        state["loss"] = loss
        torch.save(state, os.path.join(self.args.ckpt_dir, f"epoch{self.epoch}.pth"))
        logging.info(f"parameters of epoch {self.epoch} saved")

    def resume(self, weight_path: str):
        weight = torch.load(weight_path, map_location=self.device,)
        self.net_G.load_state_dict(weight["net_G"])
        self.epoch = weight["epoch"]
        self.global_step = weight["global_step"]


class VAE(nn.Module):
    def __init__(self, args) -> None:
        super(VAE, self).__init__()
        self.global_step = 0

    def reparameterize(cls, mu, logvar):
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_scheduler(self):
        return max((self.global_step // 10) % 10000 * 0.0001, 0.0001)

    def kl_divergence(cls, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1,))

    def step(self):
        self.global_step += 1


class Audio_Enc(VAE):
    def __init__(self, args):
        super(Audio_Enc, self).__init__(args)
        self.args = args

        self.TCN = ConvNet(
            args.audio_size, [128, 128, 96, 96, 64], dropout=args.dropout
        )
        self.share_mean = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, args.audio_hidden_size),
        )
        self.share_var = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, args.audio_hidden_size),
        )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        output = self.TCN(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        if self.args.with_audio_share_vae:
            self.z_share_mu = self.share_mean(output)
            self.z_share_var = self.share_var(output)
            z_share = self.reparameterize(self.z_share_mu, self.z_share_var)
        else:
            z_share = self.share_mean(output)
        return z_share

    def get_loss_dict(self):
        loss_dict = {}
        if self.args.with_audio_share_vae:
            loss_dict.update(
                {
                    "KL/audio_share": self.kl_divergence(
                        self.z_share_mu, self.z_share_var
                    )
                    * self.args.lambda_kl
                    * self.kl_scheduler(),
                }
            )
        self.step()
        return loss_dict


class Motion_Enc(VAE):
    def __init__(self, args):
        super(Motion_Enc, self).__init__(args)
        self.args = args
        input_channel = args.input_joint_num * args.input_joint_repr_dim
        self.TCN = ConvNet(
            input_channel, [256, 256, 128, 128, 64], dropout=args.dropout,
        )
        self.share_linear = nn.Linear(64, 32)
        self.spec_linear = nn.Linear(64, 32)
        self.share_mean = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, args.pose_hidden_size),
        )
        self.share_var = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, args.pose_hidden_size),
        )
        self.spec_mean = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, args.pose_hidden_size),
        )
        self.spec_var = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, args.pose_hidden_size),
        )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        output = self.TCN(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        share_output = self.share_linear(output)
        spec_output = self.spec_linear(output)

        if self.args.with_motion_share_vae:
            self.z_share_mu = self.share_mean(share_output)
            self.z_share_var = self.share_var(share_output)
            z_share = self.reparameterize(self.z_share_mu, self.z_share_var)
        else:
            z_share = self.share_mean(share_output)
        if self.args.with_motion_spec_vae:
            self.z_spec_mu = self.spec_mean(spec_output)
            self.z_spec_var = self.spec_var(spec_output)
            z_specific = self.reparameterize(self.z_spec_mu, self.z_spec_var)

        else:
            z_specific = self.spec_mean(spec_output)

        return z_share, z_specific

    def get_loss_dict(self):
        loss_dict = {}
        if self.args.with_motion_share_vae:
            loss_dict.update(
                {
                    "KL/motion_share": self.kl_divergence(
                        self.z_share_mu, self.z_share_var
                    )
                    * self.args.lambda_kl
                    * self.kl_scheduler(),
                }
            )
        if self.args.with_motion_spec_vae:
            loss_dict.update(
                {
                    "KL/motion_spec": self.kl_divergence(
                        self.z_spec_mu, self.z_spec_var
                    )
                    * self.args.lambda_kl
                    * self.kl_scheduler(),
                }
            )
        self.step()
        return loss_dict


class Motion_Dec(VAE):
    def __init__(self, args):
        super(Motion_Dec, self).__init__(args)
        self.args = args
        output_dim = args.output_joint_repr_dim * args.output_joint_num

        self.TCN = ConvNet(
            args.hidden_size, [64, 128, 128, 256, 256,], dropout=args.dropout,
        )
        self.pose_g = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, output_dim),
        )

    def forward(self, share_feature: torch.Tensor, spec_feature: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        output = torch.cat((share_feature, spec_feature), dim=2)
        output = self.TCN(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.pose_g(output)
        return output



class MappingNet(VAE):
    def __init__(self, args):
        super(MappingNet, self).__init__(args)
        self.args = args
        hidden_size = args.pose_hidden_size
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )
        self.spec_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.spec_var = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, inputs: torch.Tensor):
        output = self.net(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        if self.args.with_mapping_net_vae:
            self.z_spec_mu = self.spec_mean(output)
            self.z_spec_var = self.spec_var(output)
            z_specific = self.reparameterize(self.z_spec_mu, self.z_spec_var)
        else:
            z_specific = self.spec_mean(output)
        return z_specific

    def get_loss_dict(self):
        loss_dict = {}
        if self.args.with_mapping_net_vae:
            loss_dict.update(
                {
                    "KL/Mapping": self.kl_divergence(self.z_spec_mu, self.z_spec_var)
                    * self.args.lambda_kl
                    * self.kl_scheduler(),
                }
            )
        self.step()
        return loss_dict
