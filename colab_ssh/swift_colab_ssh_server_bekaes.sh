# creating outgoing ssh tunnel from colab vm to external server 
# exposing colab's ssh port 22 on it as port 8888
# inputs:
# - SSH_HOST env var
# - /content/authorized_keys
# - /content/private_key.pem

baseURL=https://raw.githubusercontent.com/wojtekcz/poetry2021/master/colab_ssh/

# setup utilities
apt-get update && apt-get install -y autossh tmux mc htop
pip2 install glances google-auth-oauthlib==0.4.1 grpcio==1.24.3
wget -nv $baseURL/bashrc -O /root/.bashrc

# setup ssh
mkdir -p --mode=700 /root/.ssh
authorizedKeysPath=/root/.ssh/authorized_keys
privateKeyPath=/root/.ssh/private_key.pem
cp /content/authorized_keys $authorizedKeysPath && chmod 600 $authorizedKeysPath
cp /content/private_key.pem $privateKeyPath && chmod 600 $privateKeyPath

apt-get install -y openssh-server
wget -nv $baseURL/sshd_config -O /etc/ssh/sshd_config
/etc/init.d/ssh restart
/etc/init.d/ssh status

options="-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null"
ssh $options -i $privateKeyPath -f -R 8888:localhost:22 $SSH_HOST -N -v &
