DSET_DIR="/content/"
CKPT_DIR="/content/drive/MyDrive/Saved_models/ContrastiveDisentanglement/simclr-celeba"
LOG_DIR="/content/drive/MyDrive/Saved_logs/ContrastiveDisentanglement/simclr-celeba"

python src/simclr/main.py \
    --n_epochs 200 \
    --batch_size 256 \
    --img_sz 128 \
    --ckpt_dir $CKPT_DIR \
    --dset_dir $DSET_DIR \
    --log_dir $LOG_DIR \
    --ckpt_save_epoch 50 \
    --base_encoder resnet18 \
    --dataset CelebA
