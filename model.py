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


class SMPLXModel(nn.Module):
    def __init__(self, smpl_path) -> None:
        super(SMPLXModel, self).__init__()
        data_struct = load_smplx_model(smpl_path)
        self.register_buffer("parents", data_struct["parents"])
        self.joint_num = len(self.parents)
        if len(self.parents) == 55:
            meta_betas = torch.zeros(1, 20).float()
        else:
            meta_betas = torch.zeros(1, 10).float()
        v_shaped = data_struct["v_template"] + blend_shapes(
            meta_betas, data_struct["shapedirs"]
        )
        self.register_buffer("J", vertices2joints(data_struct["J_regressor"], v_shaped))

    def forward(self, rot_mats: torch.Tensor):
        if rot_mats.shape[1] == 22:
            rot_mats = torch.cat(
                (
                    rot_mats,
                    torch.eye(3)
                    .repeat(rot_mats.shape[0], 33, 1, 1)
                    .to(rot_mats.device),
                ),
                axis=1,
            )
        poses, _ = batch_rigid_transform(
            rot_mats,
            self.J.expand(rot_mats.shape[0], self.joint_num, 3),
            self.parents,
            dtype=torch.float32,
        )
        return poses


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


class Process_3D_Motion(Motion_Process):
    def __init__(self, args) -> None:
        super(Process_3D_Motion, self).__init__()
        self.smpl_model = SMPLXModel(args.smpl_path)
        self.with_translation = args.with_translation
        self.seq_len = args.seq_len

    def encode_motion(self, motion: torch.Tensor, trans=None) -> torch.Tensor:
        B, T = motion.shape[:2]

        motion = motion.reshape(B, T, -1)
        if trans is not None:
            motion = torch.cat((motion, trans), dim=-1)
        return motion

    def decode_motion(self, motion: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        B, T = motion.shape[:2]
        if self.with_translation:
            rot_mats = motion[:, :, :-3]
            rot_mats = rot_mats.reshape(B, T, -1, 3, 3)
            trans = motion[:, :, -3:]
            return rot_mats, trans
        else:
            rot_mats = motion
            return rot_mats

    def calculate_pos(self, motion: torch.Tensor, trans=None) -> torch.Tensor:
        B, T = motion.shape[:2]
        poses = self.smpl_model(motion.reshape(B * T, -1, 3, 3)).reshape(B, T, -1, 3)
        if trans is not None:
            poses = poses + trans.unsqueeze(2)
        return poses


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
        self.feat_in_time_domain = args.feat_in_time_domain
        self.freqbasis = args.freqbasis
        if args.dataset == "Speech2Gestures":
            self.motion_processor = Process_S2G_Motion(args)
        elif args.dataset == "Trinity":
            self.motion_processor = Process_3D_Motion(args)
        else:
            raise NotImplementedError

        self.smpl_model = SMPLXModel(args.smpl_path)

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
            z_x = normal.sample((self.args.seq_len,)).permute(1, 0, 2)
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
                    mean=means[idx : idx + 1], var=vars[idx : idx + 1],
                )
            else:
                z_motion_spec = self.sampling(size=z_audio_share.shape)
        else:
            _, z_motion_spec = self.net_G["motion_enc"](motions[:, :seq_len])
        pred_motions = self.net_G["motion_dec"].inference(z_audio_share, z_motion_spec)
        return pred_motions

    def train_one_batch(self, audios: torch.Tensor, motions: torch.Tensor):
        self.z_audio_share = self.net_G["audio_enc"](audios)
        (self.z_motion_share, self.z_motion_specific,) = self.net_G["motion_enc"](
            motions
        )

        recon_m = self.net_G["motion_dec"](self.z_motion_share, self.z_motion_specific)
        a2m = self.net_G["motion_dec"](self.z_audio_share, self.z_motion_specific)
        self.z_x = self.sampling(
            size=self.z_motion_specific.shape,
            mean=self.z_motion_specific.mean(dim=(1,)),
            var=self.z_motion_specific.std(dim=(1,)),
        )
        z_x2 = self.sampling(
            size=self.z_motion_specific.shape,
            mean=self.z_motion_specific.mean(dim=(1,)),
            var=self.z_motion_specific.std(dim=(1,)),
        )

        a2x = self.net_G["motion_dec"](self.z_audio_share, self.z_x)
        a2x2 = self.net_G["motion_dec"](self.z_audio_share, z_x2)
        if self.args.with_translation:
            a2x, a2x_trans = self.motion_processor.decode_motion(a2x)
            (_, self.z_a2x_spec) = self.net_G["motion_enc"](
                self.motion_processor.encode_motion(a2x, a2x_trans)
            )
        else:
            a2x = self.motion_processor.decode_motion(a2x)
            (_, self.z_a2x_spec) = self.net_G["motion_enc"](
                self.motion_processor.encode_motion(a2x)
            )
        return recon_m, a2m, a2x, a2x2

    def calculate_2d_loss(self, tgt_p, recon_p, a2m_p, a2x_p, a2x2_p, batch):
        tgt_p = self.motion_processor.decode_motion(tgt_p)
        recon_p = self.motion_processor.decode_motion(recon_p)
        a2m_p = self.motion_processor.decode_motion(a2m_p)
        a2x_p = self.motion_processor.decode_motion(a2x_p)
        a2x2_p = self.motion_processor.decode_motion(a2x2_p)

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
            * self.args.lambda_pose,
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
                    "code/cyc": F.l1_loss(self.z_a2x_spec, self.z_x)
                    * self.args.lambda_cyc
                }
            )
        if self.args.with_ds:
            loss_G_dict.update(
                {
                    "pos/diverse": -F.l1_loss(a2x_p, a2x2_p.detach())
                    * self.args.lambda_ds,
                }
            )
        loss_G_dict.update(self.net_G["audio_enc"].get_loss_dict())
        loss_G_dict.update(self.net_G["motion_enc"].get_loss_dict())
        loss_G_dict.update(self.net_G["mapping_net"].get_loss_dict())
        self.log(batch, loss_G_dict)
        loss_G = torch.stack(list(loss_G_dict.values())).sum()
        return loss_G

    def calculate_3d_loss(self, tgt_r, tgt_t, recon_m, a2m, a2x, a2x2, batch):
        # tgt_r, tgt_t = self.motion_processor.decode_motion(tgt_m)
        recon_r, recon_t = self.motion_processor.decode_motion(recon_m)
        a2m_r, a2m_t = self.motion_processor.decode_motion(a2m)
        a2x_r, a2x_t = self.motion_processor.decode_motion(a2x)
        a2x2_r, a2x2_t = self.motion_processor.decode_motion(a2x2)

        tgt_p = self.motion_processor.calculate_pos(tgt_r, tgt_t)
        recon_p = self.motion_processor.calculate_pos(recon_r, recon_t)
        a2m_p = self.motion_processor.calculate_pos(a2m_r, a2m_t)
        a2x_p = self.motion_processor.calculate_pos(a2x_r, a2x_t)
        a2x2_p = self.motion_processor.calculate_pos(a2x2_r, a2x2_t)

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
            * self.args.lambda_pose,
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
                    "code/cyc": F.l1_loss(self.z_a2x_spec, self.z_x)
                    * self.args.lambda_cyc
                }
            )
        if self.args.with_ds:
            loss_G_dict.update(
                {
                    "pos/diverse": -F.l1_loss(a2x_p, a2x2_p.detach())
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
        if not os.path.exists(self.args.ckpt_dir):
            os.mkdir(self.args.ckpt_dir)
        else:
            if len(os.listdir(self.args.ckpt_dir)) > 0:
                logging.warning("ckpt dir not empty")
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
                if self.args.with_translation:
                    trans = data["trans"].float().to(self.device)
                self.optimG.zero_grad()

                src_motion = motion.clone()
                tgt_motion = motion.clone()
                if self.args.with_translation:
                    src_trans = trans.clone()
                    tgt_trans = trans.clone()
                    recon_m, a2m, a2x, a2x2 = self.train_one_batch(
                        audios,
                        self.motion_processor.encode_motion(src_motion, src_trans),
                    )
                else:
                    recon_m, a2m, a2x, a2x2 = self.train_one_batch(
                        audios, self.motion_processor.encode_motion(src_motion),
                    )
                if self.args.using_2D_data:
                    loss_G = self.calculate_2d_loss(
                        tgt_motion, recon_m, a2m, a2x, a2x2, batch
                    )
                else:
                    loss_G = self.calculate_3d_loss(
                        tgt_motion, tgt_trans, recon_m, a2m, a2x, a2x2, batch
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
        self.freqbasis = args.freqbasis
        self.feat_in_time_domain = args.feat_in_time_domain

    def reparameterize(cls, mu, logvar):
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_scheduler(self):
        return max((self.global_step // 10) % 10000 * 0.0001, 0.0001)

    def kl_divergence(cls, mu, var):
        return torch.mean(-0.5 * torch.sum(1 + var - mu.pow(2) - var.exp(), dim=2,))

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
        if self.args.using_2D_data:
            joint_repr_dim = 2
        else:
            joint_repr_dim = 9
        input_channel = args.joint_num * joint_repr_dim
        if self.args.with_translation:
            input_channel += 3
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
        if args.using_2D_data:
            joint_repr_dim = 2
        else:
            joint_repr_dim = 6
        output_dim = joint_repr_dim * self.args.joint_num
        if self.args.with_translation:
            output_dim += 3

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
        if not self.args.using_2D_data:
            B, T, _ = output.shape
            if self.args.with_translation:
                rot = output[:, :, :-3]
                trans = output[:, :, -3:]
            else:
                rot = output
            rot = rotation_6d_to_matrix(rot.reshape(-1, 6)).reshape(B, T, -1, 3, 3)
            if self.args.with_translation:
                output = torch.cat((rot.reshape(B, T, -1), trans), dim=-1)
            else:
                output = rot
        return output

    def inference(self, z_share, z_spec):
        z = torch.cat((z_share, z_spec), dim=2)
        output = self.TCN(z.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.pose_g(output)
        if not self.args.using_2D_data:
            B, T, _ = output.shape
            if self.args.with_translation:
                rot = output[:, :, :-3]
                trans = output[:, :, -3:]
            else:
                rot = output
            rot = rotation_6d_to_matrix(rot.reshape(-1, 6)).reshape(B, T, -1, 3, 3)
            if self.args.with_translation:
                output = torch.cat((rot.reshape(B, T, -1), trans), dim=-1)
            else:
                output = rot
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
