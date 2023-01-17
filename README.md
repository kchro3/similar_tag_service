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

## End-to-end benchmarking

We can benchmark the service and evaluate the end-to-end results with a livetest.

```commandline
➜ docker run --rm -t -p 8000:80 similar-tag-service
➜ python test/livetest/main.py \
    --input_data_path=../../notebooks/data/cotag/gold.jsonl \
    --port=8000 \
    --qps=50 \
    --max_requests=100 \
    --downsample=100

100%|██████████| 100/100 [00:11<00:00,  8.76it/s]

Latency (ms)

    p50: 8.41ms
    p75: 9.22ms
    p90: 10.12ms
    p95: 10.85ms
    p99: 12.95ms

Metrics:
 - Average jaccard: 0.00
 - think of better metrics...
```
