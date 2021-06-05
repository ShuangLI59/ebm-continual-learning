CUDA_VISIBLE_DEVICES=0 python main.py \
--experiment splitMNIST \
--scenario class \
--lr 1e-4 \
--seed 0 \
--pdf \
--epc-per-virtual-task 10 \
--save-dir splitMNIST/boundary-agnostic/ebm-class \
--ebm \

