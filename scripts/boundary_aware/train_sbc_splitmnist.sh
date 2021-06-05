CUDA_VISIBLE_DEVICES=1 python main.py \
--experiment splitMNIST \
--scenario class \
--iters 2000 \
--lr 1e-4 \
--seed 0 \
--pdf \
--task-boundary \
--save-dir splitMNIST/boundary-aware/sbc-class-standard1 \
--cls-standard 1 \

