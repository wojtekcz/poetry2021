# docker-compose commands
docker-compose build

## with jupyter lab
docker-compose up

## bash in running container
docker-compose exec conda bash

## bash in new container
docker-compose run conda bash

## cleanups
docker-compose down

# operations
## download dataset
./setup.sh

## run preprocessing
python preprocess_dataset.py

## run training
python roberta_train_script.py

## run eval
python evaluate.py
