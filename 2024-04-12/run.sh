# python3 main.py --pretrained False \
#                 --optimizer 'SGD' \
#                 --batch_size 32 \
#                 --learning_rate 1e-1 \
#                 --lr_scheduler 'Step' \
#                 --step_size 20 \
#                 --gamma 0.2 \
#                 --epoch 50 \
#                 --save 'ckpt_sgd' \
#                 >> './result/convnext.txt'

# python3 main.py --pretrained False \
#                 --optimizer 'Adam' \
#                 --batch_size 64 \
#                 --weight_decay 1e-3 \
#                 --learning_rate 1e-3 \
#                 --epoch 50 \
#                 --save 'ckpt_adam' \
#                 >> './result/convnext.txt'

python3 main.py --pretrained False \
                --optimizer 'SGD' \
                --batch_size 32 \
                --learning_rate 1e-1 \
                --epoch 50 \
                --save 'ckpt_sgd' \
                >> './result/convnext.txt'

python3 main.py --pretrained False \
                --optimizer 'SGD' \
                --batch_size 32 \
                --learning_rate 1e-2 \
                --epoch 50 \
                --save 'ckpt_sgd' \
                >> './result/convnext.txt'