torchrun --nproc_per_node=4 train.py --data_path ../datasets/CIFAR-10 \
                                     --distribution True \
                                     --batch_size 256 \
                                     --learning_rate 0.001 \
                                     --epoch 30 \
                                     > ./result/ddp.txt