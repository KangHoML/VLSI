python3 train.py --batch_size 16 \
                 --learning_rate 1e-4 \
                 --epoch 20 \
                 --save resnet_ae \
                 > ./result/loss.txt

python3 test.py --batch_size 16 \
                --sample 5 \
                --load resnet_ae
                