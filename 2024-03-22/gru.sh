torchrun --nproc_per_node 4 train.py --is_ddp True \
                                     --is_amp False \
                                     --model 'gru' \
                                     --hidden_size 512 \
                                     --embed_size 256 \
                                     --n_layers 1 \
                                     --step_size 5 \
                                     --gamma 0.1 \
                                     --batch_size 32 \
                                     --learning_rate 1e-2 \
                                     --epoch 10 \
                                     --save "gru_ddp" \
                                     > ./result/gru.txt

python3 train.py --model 'gru' \
                 --hidden_size 512 \
                 --embed_size 256 \
                 --n_layers 1 \
                 --step_size 3 \
                 --gamma 0.1 \
                 --batch_size 64 \
                 --learning_rate 1e-2 \
                 --epoch 10 \
                 --save "gru" \
                 > ./result/gru.txt
