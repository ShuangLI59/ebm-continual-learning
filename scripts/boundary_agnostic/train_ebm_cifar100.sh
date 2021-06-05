CUDA_VISIBLE_DEVICES=3 python main.py \
--experiment cifar100 \
--scenario class \
--lr 1e-4 \
--seed 0 \
--pdf \
--epc-per-virtual-task 50 \
--save-dir cifar100/boundary-agnostic/ebm-class \
--ebm \


