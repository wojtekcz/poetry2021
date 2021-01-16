# docker setup
## download dataset
dataset_path=/workspace/data/pan_tadeusz
mkdir -p $dataset_path
wget --no-clobber -P $dataset_path https://github.com/wojtekcz/poetry2021/releases/download/v0.1/pan_tadeusz.txt

# colab setup

## ssh tunnel
# in notebook:
# - upload private_key.pem and authorized_keys to /content
# - run: !SSH_RELAY_HOST=<user>@<host> SSH_RELAY_PORT=<port> <(curl -s https://raw.githubusercontent.com/wojtekcz/poetry2021/master/colab_ssh/swift_colab_ssh_server.sh)

## setup stemmer
cd /usr/local
wget --no-clobber -q https://github.com/wojtekcz/poetry2021/releases/download/v0.1/stemmer-2.0.3.tgz
tar xzf stemmer-2.0.3.tgz
rm stemmer-2.0.3.tgz

## setup python libs
pip3 install sentencepiece nlp transformers==4.0.1
pip3 install tokenizers==0.10.0rc1

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
