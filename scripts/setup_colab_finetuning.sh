# colab setup

## setup stemmer
cd /usr/local
wget --no-clobber -q https://github.com/wojtekcz/poetry2021/releases/download/v0.1/stemmer-2.0.3.tgz
tar xzf stemmer-2.0.3.tgz
rm stemmer-2.0.3.tgz

## setup python libs
pip3 install sentencepiece nlp transformers==4.2.1
pip3 install tokenizers==0.9.4 datasets==1.2.1

cd /content
git clone https://github.com/huggingface/transformers.git
cd transformers
pip3 install -e .
cd /content

# setup git
git config --global user.name "Wojtek Czarnowski"
git config --global user.email "wojtek.czarnowski@gmail.com"

## download dataset
dataset_path=/content/poetry2021/data/pan_tadeusz7
mkdir -p $dataset_path
cd $dataset_path
wget --no-clobber -P $dataset_path https://github.com/wojtekcz/poetry2021/releases/download/v0.2/pan_tadeusz.ds7.tgz
tar xzf pan_tadeusz.ds7.tgz

## download language model
lm_path=/content/poetry2021/data/pan_tadeusz7_lm_models
mkdir -p $lm_path
cd $lm_path
wget --no-clobber -P $lm_path https://github.com/wojtekcz/poetry2021/releases/download/v0.2/model1000.tgz
tar xzf model1000.tgz
