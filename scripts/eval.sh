source /root/miniconda3/bin/activate grasp

python eval_cli.py \
    --load-checkpoint-dir checkpoints/bbox_mlp/epoch2_step25695 \
    --test-split all \
    --use-bbox \
    --action-head MLP \
    --visualize-dir visualize/bbox_mlp \
    --result-dir results/bbox_mlp \
    --device cuda:0 \
