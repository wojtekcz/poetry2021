# create outgoing ssh tunnel from Colab VM to external server 
# expose Colab's ssh-server port 22 on it as port SSH_RELAY_PORT
# inputs:
# - SSH_RELAY_HOST, SSH_RELAY_PORT env vars
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
ssh $options -i $privateKeyPath -f -R $SSH_RELAY_PORT:localhost:22 $SSH_RELAY_HOST -N -v &
