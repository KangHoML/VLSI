# torchrun --nproc_per_node 4 train.py --is_ddp True \
#                                      --is_amp False \
#                                      --tokenizer "nltk" \
#                                      --vocab_size 10000 \
#                                      --model 'rnn' \
#                                      --hidden_size 128 \
#                                      --embed_size 100 \
#                                      --n_layers 1 \
#                                      --step_size 5 \
#                                      --gamma 1.0 \
#                                      --batch_size 64 \
#                                      --learning_rate 1e-4 \
#                                      --epoch 10 \
#                                      --save "rnn_ddp" \
#                                      > ./result/rnn.txt

# torchrun --nproc_per_node 4 train.py --is_ddp True \
#                                      --is_amp False \
#                                      --tokenizer "nltk" \
#                                      --vocab_size 40000 \
#                                      --model 'rnn' \
#                                      --bidirectional True \
#                                      --hidden_size 256 \
#                                      --embed_size 128 \
#  				                     --dropout 0.2 \
#                                      --n_layers 1 \
#                                      --step_size 7 \
#                                      --gamma 1.0 \
#                                      --batch_size 64 \
#                                      --learning_rate 1e-3 \
#                                      --epoch 10 \
#                                      --save "bi_rnn_ddp" \
#                                      >> ./result/rnn.txt

torchrun --nproc_per_node 4 train.py --is_ddp True \
                               	     --is_amp False \
                                     --tokenizer "nltk" \
                                     --vocab_size 40000 \
                                     --model 'rnn' \
                                     --hidden_size 256 \
                                     --embed_size 128 \
                                     --n_layers 3 \
   				                     --dropout 0.5 \
				                     --step_size 5 \
                                     --gamma 0.1 \
                                     --batch_size 64 \
                                     --learning_rate 1e-2 \
                                     --epoch 20 \
                                     --save "deep_rnn_ddp" \
                                     >> ./result/rnn.txt

# python3 train.py --tokenizer "nltk" \
#                  --vocab_size 40000 \
#                  --model 'rnn' \
#                  --hidden_size 512 \
#                  --embed_size 256 \
#                  --n_layers 1 \
#                  --step_size 5 \
#                  --gamma 0.1 \
#                  --batch_size 64 \
#                  --learning_rate 1e-2 \
#                  --epoch 10 \
#                  --save "rnn" \
#                  >> ./result/rnn.txt
