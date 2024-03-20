# # effects of depth in VGG 
# python3 train.py --data_path "../../datasets/CIFAR-10" \
#                  --model "VGG16" \
#                  --learning_rate 1e-4 \
#                  > ./result/VGG.txt

# python3 train.py --data_path "../../datasets/CIFAR-10" \
#                  --model "VGG19" \
#                  --learning_rate 1e-4 \
#                  >> ./result/VGG.txt

# effects of depth in ResNet
python3 train.py --data_path "../../datasets/CIFAR-10" \
                 --model "ResNet34" \
                 > ./result/ResNet.txt

python3 train.py --data_path "../../datasets/CIFAR-10" \
                 --model "ResNet50" \
                 >> ./result/ResNet.txt

python3 train.py --data_path "../../datasets/CIFAR-10" \
                 --model "ResNet101" \
                 >> ./result/ResNet.txt

# # effects of kernel_size in CustomNet
# python3 train.py --data_path "../../datasets/CIFAR-10" \
#                  --model "CustomNet" \
#                  --kernel_size 3 \
#                  --epoch 40 \
#                  > ./result/CustomNet.txt

# python3 train.py --data_path "../../datasets/CIFAR-10" \
#                  --model "CustomNet" \
#                  --kernel_size 5 \
#                  --epoch 40 \
#                  >> ./result/CustomNet.txt