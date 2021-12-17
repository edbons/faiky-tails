#!/bin/bash
# git clone https://github.com/edbons/faiky-tails.git
# cd faiky-tails

sudo apt update
sudo apt install python3-pip

pip3 install -r requirements_cuda11.txt

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1--3UleOTG5Y-HIJaL9stQ5TO_0xXLe7Y' -O dataset/plot/train_encoded.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-8h01YYnDda3gIERZl5fG2uJchpczN6u' -O dataset/plot/test_encoded.pkl 
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-7DDxLnzChFvR5ugQjB6PuFk5kcI7rV7' -O dataset/plot/val_encoded.pkl

python3 src/model/train.py --data_dir dataset/plot/ --output_dir out --experiment_name pmfull --gen_len 401 --n_embd 768 --accum_iter 4 --n_batch 2 --p 90 --num_epochs 1 --max_ex 2 --num_val_examples 2 --use_model plotmachines --use_neighbor_feat --use_discourse --show_progress