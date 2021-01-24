# vastai setup

apt install -y wget

## setup stemmer
dpkg --add-architecture i386 && apt-get -qq update && apt-get -qq install libc6:i386 libncurses5:i386 libstdc++6:i386
cd /usr/local
wget --no-clobber -q https://github.com/wojtekcz/poetry2021/releases/download/v0.1/stemmer-2.0.3.tgz
tar xzf stemmer-2.0.3.tgz
rm stemmer-2.0.3.tgz

## setup python libs
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install datasets
pip3 install tensorboard tensorflow glances

## download dataset
dataset_path=/root/poetry2021/data/pan_tadeusz
mkdir -p $dataset_path
wget --no-clobber -P $dataset_path https://github.com/wojtekcz/poetry2021/releases/download/v0.1/pan_tadeusz.txt

# setup git
git config --global user.name "Wojtek Czarnowski"
git config --global user.email "wojtek.czarnowski@gmail.com"

cd /root/poetry2021

export PYTHONIOENCODING=UTF-8 
