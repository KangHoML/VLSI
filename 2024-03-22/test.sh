python3 test.py --pth_name 'lstm_ddp' \
                --model 'lstm' \
                --hidden_size 512 \
                --embed_size 256 \
                --n_layers 1 \
                --batch_size 32 \
                >> ./result/lstm.txt

python3 test.py --pth_name 'lstm' \
                --model 'lstm' \
                --hidden_size 512 \
                --embed_size 256 \
                --n_layers 1 \
                --batch_size 32 \
                >> ./result/lstm.txt

python3 test.py --pth_name 'gru_ddp' \
                --model 'gru' \
                --hidden_size 512 \
                --embed_size 256 \
                --n_layers 1 \
                --batch_size 32 \
                >> ./result/gru.txt

python3 test.py --pth_name 'gru' \
                --model 'gru' \
                --hidden_size 512 \
                --embed_size 256 \
                --n_layers 1 \
                --batch_size 32 \
                >> ./result/gru.txt