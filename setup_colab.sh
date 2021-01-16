# colab setup

## setup stemmer
cd /usr/local
wget --no-clobber -q https://github.com/wojtekcz/poetry2021/releases/download/v0.1/stemmer-2.0.3.tgz
tar xzf stemmer-2.0.3.tgz
rm stemmer-2.0.3.tgz

## setup python libs
# pip3 install sentencepiece nlp transformers==4.0.1
# pip3 install tokenizers==0.10.0rc1

pip3 install sentencepiece nlp transformers==4.2.1
pip3 install tokenizers==0.9.4

## project sources setup
cd /content
git clone https://github.com/wojtekcz/poetry2021.git

## download dataset
dataset_path=/content/poetry2021/data/pan_tadeusz
mkdir -p $dataset_path
wget --no-clobber -P $dataset_path https://github.com/wojtekcz/poetry2021/releases/download/v0.1/pan_tadeusz.txt

# setup git
git config --global user.name "Wojtek Czarnowski"
git config --global user.email "wojtek.czarnowski@gmail.com"
