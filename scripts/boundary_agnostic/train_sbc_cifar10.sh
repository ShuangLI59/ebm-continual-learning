CUDA_VISIBLE_DEVICES=2 python main.py \
--experiment cifar10 \
--scenario class \
--lr 1e-4 \
--seed 0 \
--pdf \
--epc-per-virtual-task 50 \
--save-dir cifar10/boundary-agnostic/sbc-class-standard1 \
--cls-standard 1 \


