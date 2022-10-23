FROM python:3.9
COPY . /src
COPY ./requirements.txt /src/requirements.txt
WORKDIR src
EXPOSE 8000:8000
RUN apt-get update && apt-get install python3-pip -y
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--reload"]