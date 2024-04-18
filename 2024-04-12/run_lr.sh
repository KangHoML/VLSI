python3 main.py --pretrained False \
                --optimizer 'SGD' \
                --batch_size 32 \
                --learning_rate 1e-1 \
                --lr_scheduler 'Cosine' \
                --step_size 50 \
                --epoch 100 \
                --save 'ckpt_lr_1e-1' \
                > './result/convnext.txt'

python3 main.py --pretrained False \
                --optimizer 'SGD' \
                --batch_size 32 \
                --learning_rate 1e-2 \
                --step_size 50 \
                --epoch 100 \
                --save 'ckpt_lr_1e-2' \
                >> './result/convnext.txt'

python3 main.py --pretrained False \
                --optimizer 'SGD' \
                --batch_size 32 \
                --learning_rate 1e-3 \
                --lr_scheduler 'Cosine' \
                --step_size 50 \
                --epoch 100 \
                --save 'ckpt_lr_1e-3' \
                > './result/convnext.txt'

python3 main.py --pretrained False \
                --optimizer 'SGD' \
                --batch_size 32 \
                --learning_rate 1e-4 \
                --step_size 50 \
                --epoch 100 \
                --save 'ckpt_lr_1e-4' \
                >> './result/convnext.txt'