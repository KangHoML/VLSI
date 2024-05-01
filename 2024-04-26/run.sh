python3 train.py --batch_size 16 \
                 --learning_rate 1e-4 \
                 --epoch 30 \
                 --save ckpt1 \

python3 test.py --batch_size 16 \
                --sample 5 \
                --load ckpt1
                