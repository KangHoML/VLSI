torchrun --nproc_per_node 4 train.py --is_ddp True \
                                     --is_amp False \
                                     --model 'rnn' \
                                     --hidden_size 256 \
                                     --embed_size 256 \
                                     --n_layers 1 \
                                     --step_size 5 \
                                     --gamma 0.1 \
                                     --batch_size 64 \
                                     --learning_rate 1e-2 \
                                     --epoch 10 \
                                     --save "rnn_ddp" \
                                     > ./result/rnn.txt

torchrun --nproc_per_node 4 train.py --is_ddp True \
                                     --is_amp False \
                                     --model 'rnn' \
                                     --bidirectional True \
                                     --hidden_size 256 \
                                     --embed_size 128 \
                                     --n_layers 1 \
                                     --step_size 5 \
                                     --gamma 0.1 \
                                     --batch_size 64 \
                                     --learning_rate 1e-2 \
                                     --epoch 10 \
                                     --save "bi_rnn_ddp" \
                                     >> ./result/rnn.txt

torchrun --nproc_per_node 4 train.py --is_ddp True \
                                     --is_amp False \
                                     --model 'rnn' \
                                     --hidden_size 256 \
                                     --embed_size 128 \
                                     --n_layers 3 \
                                     --step_size 5 \
                                     --gamma 0.1 \
                                     --batch_size 64 \
                                     --learning_rate 1e-2 \
                                     --epoch 10 \
                                     --save "deep_rnn_ddp" \
                                     >> ./result/rnn.txt

python3 train.py --model 'rnn' \
                 --hidden_size 256 \
                 --embed_size 256 \
                 --n_layers 1 \
                 --step_size 5 \
                 --gamma 0.1 \
                 --batch_size 64 \
                 --learning_rate 1e-2 \
                 --epoch 10 \
                 --save "rnn" \
                 >> ./result/rnn.txt
