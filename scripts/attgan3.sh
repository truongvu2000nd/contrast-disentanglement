DSET_DIR="/content/LFW/"
CKPT_DIR="/content/drive/MyDrive/Saved_models/ContrastiveDisentanglement/simclr-attgan3"
LOG_DIR="/content/drive/MyDrive/Saved_logs/ContrastiveDisentanglement/simclr-attgan3"
ATTGAN_DIR="/content/ContrastiveDisentanglement/src/face-editing/attgan.pth"

python src/simclr/main.py \
    --n_epochs 200 \
    --batch_size 256 \
    --img_sz 128 \
    --ckpt_dir $CKPT_DIR \
    --dset_dir $DSET_DIR \
    --log_dir $LOG_DIR \
    --ckpt_save_epoch 50 \
    --attgan_transform True \
    --attgan_dir $ATTGAN_DIR \
    --h_flip True \
    --random_resized_crop True \
    --color_distort True \
