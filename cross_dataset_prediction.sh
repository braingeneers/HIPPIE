export WANDB_API_KEY="YOUROWNWANDBAPIKEY"
export PYTHONPATH="$(pwd)/hippie"
export TORCH_USE_CUDA_DSA=1

# Activate venv
source hippie_venv/bin/activate
wandb login --relogin $WANDB_API_KEY

python cross_dataset_script.py \
    --training-dataset "hull_cell_type" \
    --predict-dataset "lissberger_labeled_cell_type" \
    --config "class_decoder_source_bn_aug_reg" \
    --z_dim 20 \
    --beta 0.9 \
    --finetune-max-epochs 20 \
    --pretrain-max-epochs 100 \
    --supervised-max-epochs 10 \
    --use_balanced_sampling 

