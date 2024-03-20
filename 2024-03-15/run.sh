torchrun --nproc_per_node=4 train.py --data_path ../datasets/CIFAR-10 \
                                     --model "ResNeXt50" \
                                     > ./result/ResNeXt.txt

torchrun --nproc_per_node=4 train.py --data_path ../datasets/CIFAR-10 \
                                     --model "ResNeXt101" \
                                     >> ./result/ResNeXt.txt

torchrun --nproc_per_node=4 train.py --data_path ../datasets/CIFAR-10 \
                                     --model "ResNeXt152" \
                                     >> ./result/ResNeXt.txt