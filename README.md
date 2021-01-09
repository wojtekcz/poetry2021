conda list --explicit
conda env export --from-history > environment.yml
conda env update --file environment.yml 
docker-compose run --service-ports conda
