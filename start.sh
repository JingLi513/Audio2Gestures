NAME=debug
DATASET=Trinity
SMPLX_PATH=smplx/SMPLX_NEUTRAL.pkl
BASE_PATH=trinity
CKPT_PATH=$BASE_PATH/ckpt/$NAME
LOG_PATH=$BASE_PATH/log/$NAME
RESULT_PATH=$BASE_PATH/pred/$NAME
python3 main.py --base_path="$BASE_PATH/train" --ckpt_dir="$CKPT_PATH" --log_dir="$LOG_PATH" --with_audio_share_vae --with_motion_share_vae --with_motion_spec_vae --with_code_constrain  --lambda_kl 1e-3 --lambda_code 0.1 --lambda_pose 1 --lambda_speed 1 --lambda_xpose 1 --lambda_xspeed 5 --lr 1e-4 --batch_size 32 --seq_len 64 --dropout 0 --num_workers 1  --audio_stat speech_stat.npy --save_freq 100 --dataset $DATASET --using_mspec_stat --smplx_path $SMPLX_PATH --lambda_cyc 0.1 --lambda_ds 0.1 --with_cyc --with_ds  --with_mapping_net --with_mapping_net_vae

python3 calculate_training_stat.py --base_path="$BASE_PATH/train" --with_audio_share_vae --with_motion_share_vae --with_motion_spec_vae --with_mapping_net --with_mapping_net_vae --batch_size 1 --seq_len -1 --resume "$CKPT_PATH/epoch3000.pth" --using_mspec_stat --audio_stat speech_stat.npy --dataset $DATASET --smplx_path $SMPLX_PATH --result_path $RESULT_PATH

python3 test.py --base_path="$BASE_PATH/test" --with_audio_share_vae --with_motion_share_vae --with_motion_spec_vae --with_mapping_net --with_mapping_net_vae --batch_size 1 --seq_len -1 --resume "$CKPT_PATH/epoch3000.pth" --using_mspec_stat  --audio_stat speech_stat.npy  --dataset $DATASET --smplx_path $SMPLX_PATH --result_path $RESULT_PATH
