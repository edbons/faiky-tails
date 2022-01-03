#!/bin/bash
# git clone https://github.com/edbons/faiky-tails.git
# cd faiky-tails

sudo apt update
sudo apt install python3-pip
sudo apt install p7zip-full p7zip-rar

if [[ $1 == 'cuda11' ]]; then
    pip3 install -r requirements_cuda11.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
elif [[ $1 == 'colab' ]]; then
    pip3 install rouge
    pip3 install transformers==2.8.0
else
    pip3 install -r requirements.txt
fi

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1--3UleOTG5Y-HIJaL9stQ5TO_0xXLe7Y' -O dataset/plot/train_encoded.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-8h01YYnDda3gIERZl5fG2uJchpczN6u' -O dataset/plot/test_encoded.pkl 
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-7DDxLnzChFvR5ugQjB6PuFk5kcI7rV7' -O dataset/plot/val_encoded.pkl

# wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aNts8pOpCQ9wVs2e6X67vG3e0t66Ry-E' -O checkpoints.zip

# python3 src/model/train.py --data_dir dataset/plot/ --output_dir out --experiment_name $1 --gen_len 401 --n_embd 768 --accum_iter 4 --n_batch 2 --p 90 --num_epochs 1 --max_ex 2 --num_val_examples 2 --use_model $1 --use_neighbor_feat --use_discourse --show_progress

# # archivate output
# 7z a out.7z out

# # upload to gdrive
# sudo apt install musl
# wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_amd64.tar.gz
# tar -xzvf gdrive_2.1.1_linux_amd64.tar.gz
# sudo install gdrive /usr/local/bin/gdrive
# gdrive upload -p <ID> out.7z
# rm .gdrive/token_v2.json