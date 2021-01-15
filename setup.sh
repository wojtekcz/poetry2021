# docker setup
## download dataset
dataset_path=/workspace/data/pan_tadeusz
mkdir -p $dataset_path
wget --no-clobber -P $dataset_path https://github.com/wojtekcz/poetry2021/releases/download/v0.1/pan_tadeusz.txt

# colab setup

## ssh tunnel
# upload private_key.pem and authorized_keys to /content
# !SSH_HOST=<user>@<host> <(curl -s https://raw.githubusercontent.com/wojtekcz/poetry2021/master/colab_ssh/swift_colab_ssh_server_bekaes.sh)

## setup stemmer
cd /usr/local
wget --no-clobber -q https://github.com/wojtekcz/poetry2021/releases/download/v0.1/stemmer-2.0.3.tgz
tar xzf stemmer-2.0.3.tgz
rm stemmer-2.0.3.tgz

## project sources setup
cd /content
git clone https://github.com/wojtekcz/poetry2021.git

## download dataset
dataset_path=/content/poetry2021/data/pan_tadeusz
mkdir -p $dataset_path
wget --no-clobber -P $dataset_path https://github.com/wojtekcz/poetry2021/releases/download/v0.1/pan_tadeusz.txt
