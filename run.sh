## mesonet
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.001 --batch_size 512 --model_name mesonet --img_size 320 --epoch 30 --dataset_name FaceSwap --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.001 --batch_size 512 --model_name mesonet --img_size 320 --epoch 30 --dataset_name NeuralTextures --frame_num 32

## mesoincep
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.001 --batch_size 512 --model_name mesonet_incep --img_size 320 --epoch 30 --dataset_name FaceSwap --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.001 --batch_size 512 --model_name mesonet_incep --img_size 320 --epoch 30 --dataset_name NeuralTextures --frame_num 32

## efficient_vit
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.001 --weight_decay 0.0000001 --lr_decay step --batch_size 256 --model_name efficient_vit --img_size 224 --epoch 30 --dataset_name Deepfakes --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.001 --weight_decay 0.0000001 --lr_decay step --batch_size 256 --model_name efficient_vit --img_size 224 --epoch 30 --dataset_name Face2Face --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.001 --weight_decay 0.0000001 --lr_decay step --batch_size 256 --model_name efficient_vit --img_size 224 --epoch 30 --dataset_name FaceSwap --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.001 --weight_decay 0.0000001 --lr_decay step --batch_size 256 --model_name efficient_vit --img_size 224 --epoch 30 --dataset_name NeuralTextures --frame_num 32

## cnn_rnn
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 128 --model_name cnn_rnn --img_size 320 --epoch 30 --dataset_name Face2Face --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 128 --model_name cnn_rnn --img_size 320 --epoch 30 --dataset_name NeuralTextures --frame_num 32

## f3net
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 128 --model_name f3net --img_size 320 --epoch 30 --dataset_name FaceSwap --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 128 --model_name f3net --img_size 320 --epoch 30 --dataset_name NeuralTextures --frame_num 32

## xception
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 128 --model_name xception --img_size 320 --epoch 30 --dataset_name FaceSwap --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 128 --model_name xception --img_size 320 --epoch 30 --dataset_name Face2Face --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 128 --model_name xception --img_size 320 --epoch 30 --dataset_name NeuralTextures --frame_num 32

# vit
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 64 --model_name vit --img_size 320 --epoch 30 --dataset_name Deepfakes --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 64 --model_name vit --img_size 320 --epoch 30 --dataset_name FaceSwap --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 64 --model_name vit --img_size 320 --epoch 30 --dataset_name Face2Face --frame_num 32
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train_distribute_run.py --lr 0.0001 --batch_size 64 --model_name vit --img_size 320 --epoch 30 --dataset_name NeuralTextures --frame_num 32

# cipuc
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.0001 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name Deepfakes --frame_num 100 --lr_decay warmup --weight_decay 0.04
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.0001 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name FaceSwap --frame_num 100 --lr_decay multi --weight_decay 0.04 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.0001 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name Face2Face --frame_num 32 --lr_decay multi
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.0001 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name NeuralTextures --frame_num 32 --lr_decay multi
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.00008 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name Deepfakes --frame_num 32 --weight_decay 0.003 --lr_decay multi
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.00008 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name Deepfakes --frame_num 32 --weight_decay 0.003 --lr_decay warmup --aug True
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.00008 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name Face2Face --frame_num 32 --weight_decay 0.003 --lr_decay warmup --aug True
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.00008 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name FaceSwap --frame_num 32 --weight_decay 0.003 --lr_decay warmup --aug True
CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.00008 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name NeuralTextures --frame_num 32 --weight_decay 0.003 --lr_decay warmup --aug True
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.00008 --batch_size 32 --model_name cipucnet --img_size 320 --epoch 30 --dataset_name Deepfakes,Face2Face,FaceSwap,NeuralTextures --frame_num 32 --weight_decay 0.003 --lr_decay warmup --aug True
# CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.0001 --batch_size 128 --model_name xception --img_size 320 --epoch 30 --dataset_name Deepfakes,Face2Face,FaceSwap,NeuralTextures --frame_num 32 --weight_decay 3e-3 --lr_decay step --aug True

# test
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name xception --model_compression LQ --data_compression HQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name cnn_rnn --model_compression LQ --data_compression HQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name f3net --model_compression LQ --data_compression HQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name mesonet --model_compression LQ --data_compression HQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name mesonet_incep --model_compression LQ --data_compression HQ
# python3 test.py --model_name cipucnet --model_compression LQ --data_compression HQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name xception --compression LQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name cnn_rnn --compression LQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name f3net --compression LQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name mesonet --compression LQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name mesonet_incep --compression LQ
# python3 test.py --model_name cipucnet --model_compression HQ --data_compression HQ
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --model_name efficient_vit