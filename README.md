conda list --explicit
conda env export --from-history > environment.yml
conda env update --file environment.yml 
docker-compose run --service-ports conda
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install transformers==4.0.1 -c huggingface
pip install tokenizers==0.10.0rc1
