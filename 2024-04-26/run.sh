# # ResNet Auto Encoder Training
# python3 train.py --batch_size 16 \
#                  --learning_rate 1e-4 \
#                  --epoch 20 \
#                  --save "resnet_ae" \
#                  > "./result/loss.txt"

# python3 plot.py --sample 5 \
#                 --load "resnet_ae"

# # Clustering
# python3 cluster.py --cluster "KMeans" \
#                    --load "resnet_ae"

# python3 cluster.py --cluster "Hierarchical" \
#                    --load "resnet_ae"

# ResNet Auto Encoder Training
python3 train.py --batch_size 16 \
                 --learning_rate 1e-4 \
                 --epoch 20 \
                 --save "resnet_light" \
                 >> "./result/loss.txt"

python3 plot.py --sample 5 \
                --load "resnet_light"

# Clustering
python3 cluster.py --cluster "KMeans" \
                   --load "resnet_light"

python3 cluster.py --cluster "Hierarchical" \
                   --load "resnet_light"
