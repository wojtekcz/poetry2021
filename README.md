docker-compose build

# with jupyter lab
docker-compose up

# bash in running container
docker-compose exec conda bash

# bash in new container
docker-compose run conda bash

# download stuff
setup.sh

# run preprocessing
python preprocess_dataset.py

# run training
python roberta_train_script.py

# run eval
python evaluate.py
