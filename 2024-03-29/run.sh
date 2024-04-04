python3 train.py --tokenizer 'okt' \
                 --vocab_size 20000 \
                 --max_len 32 \
                 --learning_rate 2e-5 \
                 --save 'okt' \
                 > './result/okt.txt'

python3 test.py --tokenizer 'okt' \
                 --vocab_size 20000 \
                 --max_len 32 \
                 --pth_name 'okt' \
                 >> './result/okt.txt'

python3 train.py --tokenizer 'spm' \
                 --vocab_size 20000 \
                 --max_len 32 \
                 --learning_rate 2e-5 \
                 --save 'spm' \
                 > './result/spm.txt'

python3 test.py --tokenizer 'spm' \
                 --vocab_size 20000 \
                 --max_len 32 \
                 --pth_name 'spm' \
                 >> './result/spm.txt'