# !bash <(curl -s https://raw.githubusercontent.com/wojtekcz/language2motion/master/notebooks/Colab/swift_colab_ssh_server_bekaes.sh)


## get dataset
dataset_path=/workspace/data/pan_tadeusz
mkdir -p $dataset_path
wget --no-clobber -P $dataset_path https://github.com/wojtekcz/ml_seminars/releases/download/v0.1/pan_tadeusz.txt

## setup stemmer
# wget --no-clobber https://github.com/wojtekcz/ml_seminars/releases/download/v0.1/stemmer-2.0.3.tgz 
# tar xzf stemmer-2.0.3.tgz
# rm stemmer-2.0.3.tgz
