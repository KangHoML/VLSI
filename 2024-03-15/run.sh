torchrun --nproc_per_node=4 train.py --data_path ../../datasets/CIFAR-10 \
                                     --model "ResNeXt50" \
                                     --batch_size 128 \
                                     --learning_rate 1e-3 \
                                     --schedular True \
                                     --epoch 100 \
                                     > ./result/ResNeXt.txt

# torchrun --nproc_per_node=4 train.py --data_path ../../datasets/CIFAR-10 \
#                                      --model "ResNeXt101" \
#                                      >> ./result/ResNeXt.txt

# torchrun --nproc_per_node=4 train.py --data_path ../../datasets/CIFAR-10 \
#                                      --model "ResNeXt152" \
#                                      >> ./result/ResNeXt.txt
