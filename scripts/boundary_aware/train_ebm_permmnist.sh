CUDA_VISIBLE_DEVICES=4 python main.py \
--experiment permMNIST \
--scenario class \
--iters 2000 \
--lr 1e-4 \
--seed 0 \
--pdf \
--task-boundary \
--save-dir permMNIST/boundary-aware/ebm-class \
--ebm \



