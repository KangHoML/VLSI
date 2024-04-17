python3 main.py --pretrained False \
                --optimizer 'SGD' \
                --batch_size 32 \
                --learning_rate 1e-2 \
                --lr_scheduler 'Cosine' \
                --step_size 50 \
                --epoch 100 \
                --save 'convnext' \
                > './result/convnext.txt'