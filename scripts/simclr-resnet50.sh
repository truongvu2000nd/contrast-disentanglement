DSET_DIR="/content/LFW/"
CKPT_DIR="/content/drive/MyDrive/Saved_models/ContrastiveDisentanglement/simclr-resnet50"
LOG_DIR="/content/drive/MyDrive/Saved_logs/ContrastiveDisentanglement/simclr-resnet50"

python src/simclr/main.py \
    --n_epochs 200 \
    --batch_size 256 \
    --img_sz 128 \
    --ckpt_dir $CKPT_DIR \
    --dset_dir $DSET_DIR \
    --log_dir $LOG_DIR \
    --base_encoder resnet50 \
    --ckpt_save_epoch 50 \
