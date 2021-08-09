CUDA_VISIBLE_DEVICES=0 python main.py \
--experiment cifar10 \
--scenario class \
--iters 2000 \
--lr 1e-4 \
--seed 0 \
--pdf \
--task-boundary \
--save-dir cifar10/boundary-aware/sbc-class-standard1 \
--cls-standard 1 \
