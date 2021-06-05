CUDA_VISIBLE_DEVICES=6 python main.py \
--experiment cifar100 \
--scenario class \
--iters 5000 \
--lr 1e-4 \
--seed 0 \
--pdf \
--task-boundary \
--save-dir cifar100/boundary-aware/ebm-class \
--ebm \


