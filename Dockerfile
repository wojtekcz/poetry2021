FROM continuumio/miniconda3:latest

RUN dpkg --add-architecture i386 && apt-get -qq update && apt-get -qq install libc6:i386 libncurses5:i386 libstdc++6:i386
COPY environment.yml .
RUN conda env update --name base --file environment.yml 
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /
EXPOSE 8888
CMD [ "jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--no-browser" ]
