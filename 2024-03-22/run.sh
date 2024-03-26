torchrun --nproc_per_node 4 train.py --is_ddp True \
                                     --is_amp False \
                                     --hidden_size 256 \
                                     --embed_size 128 \
                                     --dropout 0.2 \
                                     --batch_size 64 \
                                     --learning_rate 1e-3 \
                                     --epoch 10