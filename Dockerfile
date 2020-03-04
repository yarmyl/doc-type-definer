FROM ubuntu

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install locales

ENV LANG ru_RU.UTF-8

RUN locale-gen ru_RU.UTF-8 && \
    dpkg-reconfigure locales

RUN apt update && \
    apt install -y python3 python3-pip libsm6 libxext6 \
        libxrender1 poppler-utils tesseract-ocr \
        libtesseract-dev tesseract-ocr-rus && \
    apt clean -y

COPY requirements.txt /app/

RUN pip3 install --no-cache-dir -r /app/requirements.txt

WORKDIR /app/

COPY base.yml /app/

COPY scanner.py /app/

CMD ["python3", "scanner.py", "--server"]