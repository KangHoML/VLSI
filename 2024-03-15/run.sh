torchrun --nproc_per_node=4 train.py --data_path ../datasets/CIFAR-10 \
                                     --model "ResNeXt50" \
                                     > ./result/ResNext.txt

torchrun --nproc_per_node=4 train.py --data_path ../datasets/CIFAR-10 \
                                     --model "ResNeXt101" \
                                     >> ./result/ResNext.txt

torchrun --nproc_per_node=4 train.py --data_path ../datasets/CIFAR-10 \
                                     --model "ResNeXt152" \
                                     >> ./result/ResNext.txt