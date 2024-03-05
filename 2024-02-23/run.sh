python3 train.py --data_path "../../datasets/fra_eng.txt" \
                 --model "Seq2Seq" \
                 --hidden_size 256 \
                 > ./result/Seq2Seq.txt

# python3 train.py --data_path "../../datasets/fra_eng.txt" \
#                  --model "Transformer" \
#                  --hidden_size 512 \
#                  --num_layer 6 \
#                  --max_seq_len 16 \
#                  > ./result/Transformer.txt