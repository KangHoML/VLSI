python3 train.py --tokenizer 'okt' \
                 --vocab_size 20000 \
                 --max_len 32 \
                 > './result/okt.txt'

python3 test.py --tokenizer 'okt' \
                 --vocab_size 20000 \
                 --max_len 32 \
                 >> './result/okt.txt'

python3 train.py --tokenizer 'spm' \
                 --vocab_size 20000 \
                 --max_len 32 \
                 > './result/spm.txt'

python3 test.py --tokenizer 'spm' \
                 --vocab_size 20000 \
                 --max_len 32 \
                 >> './result/spm.txt'