# Similar Tag Service

author: @kchro3 (Jeff Hara)

## Running locally

I'm using PyCharm CE. In my IDE Terminal, I can run this command to spin up a local web server running 
on `http://localhost:8000`:

```commandline
➜ uvicorn src.main:app --reload    
```

To run the service in Docker, I can run these commands:

```commandline
➜ docker build -t similar-tag-service .
➜ docker run --rm -t -p 8000:80 similar-tag-service
```

## Testing locally

TODO: Need to remember how pytest works. Maybe there are other testing frameworks to look into.

## Data exploration

I'm using Jupyter lab to investigate the datasets.

```commandline
➜ cd notebooks
➜ jupyter lab
```
