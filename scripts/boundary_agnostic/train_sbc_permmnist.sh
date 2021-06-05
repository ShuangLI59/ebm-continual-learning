CUDA_VISIBLE_DEVICES=1 python main.py \
--experiment permMNIST \
--scenario class \
--lr 1e-4 \
--seed 0 \
--pdf \
--epc-per-virtual-task 10 \
--save-dir permMNIST/boundary-agnostic/sbc-class-standard1 \
--cls-standard 1 \


