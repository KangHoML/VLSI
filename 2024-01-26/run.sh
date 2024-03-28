# Tanh Activation Function
python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "Tanh" \
                 --learning_rate 1e-2 \
                 > ./result/Tanh.txt

python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "Tanh" \
                 --learning_rate 1e-3 \
                 >> ./result/Tanh.txt

python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "Tanh" \
                 --learning_rate 0.0001 \
                 >> ./result/Tanh.txt

# ReLU Activation Function
python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "ReLU" \
                 --learning_rate 1e-2 \
                 > ./result/ReLU.txt


python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "ReLU" \
                 --learning_rate 1e-3 \
                 >> ./result/ReLU.txt

python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "ReLU" \
                 --learning_rate 1e-4 \
                 >> ./result/ReLU.txt

# ELU Activation Function
python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "ELU" \
                 --learning_rate 1e-2 \
                 > ./result/ELU.txt

python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "ELU" \
                 --learning_rate 1e-3 \
                 >> ./result/ELU.txt

python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "ELU" \
                 --learning_rate 1e-4 \
                 >> ./result/ELU.txt

# SiLU Activation Function
python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "SiLU" \
                 --learning_rate 1e-2 \
                 > ./result/SiLU.txt

python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "SiLU" \
                 --learning_rate 1e-3 \
                 >> ./result/SiLU.txt

python3 train.py --data_path "../../datasets/MNIST" \
                 --act_func "SiLU" \
                 --learning_rate 1e-4 \
                 >> ./result/SiLU.txt