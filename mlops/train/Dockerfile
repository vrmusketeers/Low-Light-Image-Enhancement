FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3

WORKDIR /

COPY lol_dataset lol_dataset
COPY core /core
COPY core/utils core/utils
    
WORKDIR /core

ENTRYPOINT ["python","-m","trainer.train"]