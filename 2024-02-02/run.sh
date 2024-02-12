# effects of depth in VGG 
python3 train.py --data_path "../../datasets/CIFAR-10" \
                 --model "VGG16" \
                 > ./result/VGG.txt

python3 train.py --data_path "../../datasets/CIFAR-10" \
                 --model "VGG19" \
                 >> ./result/VGG.txt

# effects of depth in ResNet
python3 train.py --data_path "../../datasets/CIFAR-10" \
                 --model "ResNet18" \
                 > ./result/ResNet.txt

python3 train.py --data_path "../../datasets/CIFAR-10" \
                 --model "ResNet34" \
                 >> ./result/ResNet.txt

python3 train.py --data_path "../../datasets/CIFAR-10" \
                 --model "ResNet50" \
                 >> ./result/ResNet.txt