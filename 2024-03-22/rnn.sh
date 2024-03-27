# torchrun --nproc_per_node 4 train.py --is_ddp True \
#                                      --is_amp False \
#                                      --model 'rnn' \
#                                      --hidden_size 512 \
#                                      --embed_size 256 \
#                                      --n_layers 1 \
#                                      --step_size 5 \
#                                      --gamma 1.0 \
#                                      --batch_size 32 \
#                                      --learning_rate 1e-2 \
#                                      --epoch 10 \
#                                      --save "rnn_ddp" \
#                                      > ./result/rnn.txt

# torchrun --nproc_per_node 4 train.py --is_ddp True \
#                                      --is_amp False \
#                                      --model 'rnn' \
#                                      --bidirectional True \
#                                      --hidden_size 512 \
#                                      --embed_size 256 \
#  				       --dropout 0.2 \
#                                      --n_layers 1 \
#                                      --step_size 7 \
#                                      --gamma 0.1 \
#                                      --batch_size 32 \
#                                      --learning_rate 1e-2 \
#                                      --epoch 10 \
#                                      --save "bi_rnn_ddp" \
#                                      >> ./result/rnn.txt

torchrun --nproc_per_node 4 train.py --is_ddp True \
                               	     --is_amp False \
                                     --model 'rnn' \
                                     --hidden_size 512 \
                                     --embed_size 256 \
                                     --n_layers 2 \
   				     --dropout 0.2 \
				     --step_size 5 \
                                     --gamma 1.0 \
                                     --batch_size 32 \
                                     --learning_rate 1e-2 \
                                     --epoch 10 \
                                     --save "deep_rnn_ddp" \
                                     >> ./result/rnn.txt

python3 train.py --model 'rnn' \
                 --hidden_size 512 \
                 --embed_size 256 \
                 --n_layers 1 \
                 --step_size 5 \
                 --gamma 0.1 \
                 --batch_size 64 \
                 --learning_rate 1e-2 \
                 --epoch 10 \
                 --save "rnn" \
                 >> ./result/rnn.txt
