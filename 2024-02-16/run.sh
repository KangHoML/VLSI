python3 baseline.py --data_path "../../datasets/CIFAR-10" \
                    > './result/result.txt'

python3 output_logit.py --data_path "../../datasets/CIFAR-10" \
                        >> './result/result.txt'