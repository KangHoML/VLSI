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

python3 main.py --pretrained False \
                --optimizer 'Adam' \
                --batch_size 64 \
                --weight_decay 1e-3 \
                --learning_rate 1e-3 \
                --epoch 100 \
                --save 'ckpt_adam' \
                >> './result/convnext.txt'

python3 main.py --pretrained True \
                --cfgs '[(2, 64), (2, 128), (2, 256), (2, 512)]' \
                --patch_size 1 \
                --optimizer 'AdamW' \
                --batch_size 32 \
                --learning_rate 1e-3 \
                --lr_scheduler 'Cycle' \
                --epoch 100 \
                --save 'ckpt_fine' \
                >> './result/convnext.txt'

