DSET_DIR="/content/LFW/"
CKPT_DIR="/content/drive/MyDrive/Saved_models/ContrastiveDisentanglement/simclr2"
LOG_DIR="/content/drive/MyDrive/Saved_logs/ContrastiveDisentanglement/simclr2"
ATTGAN_DIR="/content/ContrastiveDisentanglement/src/face-editing/attgan.pth"

python src/simclr/main.py \
    --n_epochs 400 \
    --batch_size 512 \
    --img_sz 128 \
    --ckpt_dir $CKPT_DIR \
    --dset_dir $DSET_DIR \
    --log_dir $LOG_DIR \
    --ckpt_save_epoch 50 \
    --attgan_transform False \
    --h_flip True \
    --random_resized_crop True \
    --color_distort True \
