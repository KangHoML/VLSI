torchrun --nproc_per_node 4 train.py --is_ddp True \
                                     --is_amp False \
                                     --tokenizer "nltk" \
                                     --vocab_size 40000 \
                                     --model 'lstm' \
                                     --hidden_size 512 \
                                     --embed_size 256 \
                                     --n_layers 1 \
                                     --step_size 5 \
                                     --gamma 1.0 \
                                     --batch_size 32 \
                                     --learning_rate 1e-2 \
                                     --epoch 10 \
                                     --save "lstm_ddp" \
                                     > ./result/lstm.txt

python3 train.py --tokenizer "nltk" \
                 --vocab_size 40000 \
                 --model 'lstm' \
                 --hidden_size 512 \
                 --embed_size 256 \
                 --n_layers 1 \
                 --step_size 7 \
                 --gamma 0.1 \
                 --batch_size 64 \
                 --learning_rate 1e-2 \
                 --epoch 10 \
                 --save "lstm" \
		 >> ./result/lstm.txt