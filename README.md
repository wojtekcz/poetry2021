# poetry2021

## docker commands
```
# build image
docker-compose build

# run container with jupyter lab
docker-compose up

# run container with bash
docker-compose run conda bash

# start bash in running container
docker-compose exec conda bash

# start jupyter lab
jupyter lab --allow-root --ip=0.0.0.0 --no-browser

# cleanups
docker-compose stop
docker-compose down
```

## operations
```
# download dataset
./setup_docker.sh

# preprocessing
python3 preprocess_dataset.py

# training
python3 roberta_train_script.py

# tensorboard
tensorboard --bind_all --logdir runs/

# run eval
python3 evaluate.py
```

## colab
```
ssh-keygen -y -f private_key.pem > authorized_keys
```

### ssh tunnel
in notebook:
- upload `private_key.pem` and `authorized_keys` to `/content`
- run:
```
!SSH_RELAY_HOST=<user>@<host> SSH_RELAY_PORT=<port> bash <(curl -s https://raw.githubusercontent.com/wojtekcz/poetry2021/master/colab_ssh/colab_ssh_server.sh)
```

### run ssh
```
ssh colabp
tmux -CC
```

### setup runtime env, download sources & dataset
```
cd /content
git clone https://github.com/wojtekcz/poetry2021.git
cd poetry2021
./setup_colab.sh
```

## vast.ai
image huggingface/transformers-pytorch-gpu

touch ~/.no_auto_tmux
cd /root
git clone https://github.com/wojtekcz/poetry2021.git
cd poetry2021

./setup_vastai.sh
