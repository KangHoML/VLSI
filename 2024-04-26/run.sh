# ResNet Auto Encoder -> KMeans Cluster -> TSNE Visualization
python3 train.py --batch_size 16 \
                 --learning_rate 1e-4 \
                 --epoch 20 \
                 --save "resnet_ae" \
                 > "./result/loss.txt"

python3 test.py --sample 5 \
                --load "resnet_ae"

python3 cluster.py --cluster "KMeans" \
                   --load "resnet_ae"