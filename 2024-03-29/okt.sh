python3 train.py --tokenizer 'okt' \
                 --vocab_size 20000 \
                 --max_len 32

python3 test.py --tokenizer 'okt' \
                 --vocab_size 20000 \
                 --max_len 32

python3 train.py --tokenizer 'spm' \
                 --vocab_size 20000 \
                 --max_len 32

python3 test.py --tokenizer 'spm' \
                 --vocab_size 20000 \
                 --max_len 32