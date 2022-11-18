# intended to be used with the following docker tag: nvidia/cuda_11.6.0-devel-ubuntu20.04/ssh:latest
# this script is intended to be run locally
# make sure the .ssh/config file is set up correctly:
#   Host vastai
#   HostName ssh[fill in].vast.ai
#   User root
#   Port [fill in]
#   IdentityFile ~/.ssh/id_rsa
#   IdentitiesOnly yes
#   ForwardAgent yes

# add ssh-key for forwarding (necessary for github)
ssh-add ~/.ssh/id_rsa

# make sure tmux is not set up
ssh vastai "touch ~/.no_auto_tmux"

ssh vastai "ssh-keyscan github.com >> ~/.ssh/known_hosts"

# clone the FlatNeRF repo
echo "cloning the repo"
ssh -A vastai "git clone git@github.com:martinzlocha/svox2.git"

# prepare dir for datsets
ssh vastai "mkdir datasets"

echo "installing pytorch and other python libs"
ssh vastai "cd svox2 && pip install -r requirements.txt"
