# pre-training the source model
CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/r50_deformable_detr.sh --output_dir exps/source_model --dataset_file city2foggy_source

# training the proposed method
CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/DA_r50_deformable_detr.sh --output_dir exps/ours --transform make_da_transforms --dataset_file city2foggy --checkpoint exps/source_model/checkpoint.pth


