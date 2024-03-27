# -- hyperparameter by num_layers 
torchrun --nproc_per_node 4 train.py --is_ddp True \
                                     --is_amp False \
                                     --model 'gru' \
                                     --hidden_size 256 \
                                     --embed_size 128 \
                                     --n_layers 1 \
                                     --dropout 0.2 \
                                     --step_size 5 \
                                     --gamma 0.1 \
                                     --batch_size 64 \
                                     --learning_rate 1e-2 \
                                     --epoch 10 \
                                     --save "gru_layer_1" \
                                     > ./result/gru.txt

torchrun --nproc_per_node 4 train.py --is_ddp True \
                                     --is_amp False \
                                     --model 'gru' \
                                     --hidden_size 256 \
                                     --embed_size 128 \
                                     --n_layers 2 \
                                     --dropout 0.2 \
                                     --step_size 5 \
                                     --gamma 0.1 \
                                     --batch_size 64 \
                                     --learning_rate 1e-2 \
                                     --epoch 10 \
                                     --save "gru_layer_2" \
                                     >> ./result/gru.txt

torchrun --nproc_per_node 4 train.py --is_ddp True \
                                     --is_amp False \
                                     --model 'gru' \
                                     --hidden_size 256 \
                                     --embed_size 128 \
                                     --n_layers 3 \
                                     --dropout 0.2 \
                                     --step_size 5 \
                                     --gamma 0.1 \
                                     --batch_size 64 \
                                     --learning_rate 1e-2 \
                                     --epoch 10 \
                                     --save "gru_layer_3" \
                                     >> ./result/gru.txt
