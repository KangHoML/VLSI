python3 baseline.py --data_path "../../datasets/CIFAR-10" \
                    > './result/result.txt'

# match the output logit
python3 distillation.py --data_path "../../datasets/CIFAR-10" \
                        --match "logit" \
                        --temperature 2 \
                        --kd_weight 0.25 \
                        --ce_weight 0.75 \
                        >> './result/result.txt'

# match the hidden representation
python3 distillation.py --data_path "../../datasets/CIFAR-10" \
                        --match "representation" \
                        --kd_weight 0.25 \
                        --ce_weight 0.75 \
                        >> './result/result.txt'

# match the feature map
python3 distillation.py --data_path "../../datasets/CIFAR-10" \
                        --match "feature_map" \
                        --kd_weight 0.25 \
                        --ce_weight 0.75 \
                        >> './result/result.txt'