# Dataset Exploration

I did some initial dataset exploration in the `data_exploration.ipynb`
to get a set of what the raw data itself looks like, and I got some 
feedback on my findings in P2.

I wrote a `preprocess_data.py` script to do some basic clean-ups like 
handling NaNs and converting comma-separated strings into arrays.

```commandline
python preprocess_data.py \
    --input_data_path=data/tags_on_posts_sample.csv \
    --output_data_path=data/preprocessed_data.jsonl
```

As a proxy for measuring tag similarity, I propose using the co-occurrence 
of tags in posts, so we can have a hold-out set of posts for which we cannot use
for modeling.

```commandline
python train_test_splitter.py \
   --input_data_path=data/preprocessed_data.jsonl \
   --output_data_dir=data/cotag \
   --hold_out_pct=0.1  # 10% of data is held-out
```

Another thought was that while I could write a model-agnostic evaluation harness to compare
model performance, but it's a bit of a time-sink to do from scratch.

In addition, I will want to have an end-to-end performance evaluation as well, so that if 
we compose multiple components together (e.g. multiple candidate sources, filtering, ranking),
we can see how they improve performance holistically.

Therefore, we can kill two birds with one stone by writing an end-to-end benchmark test
that takes in a set of input tags and expected output tags, using our hold-out set, and
we can feed this dataset to a running instance of our service. This can double as a kind 
of smoketest + benchmarking test for end-to-end performance.

I would probably not do this in production because we would have A/B testing, but for 
a prototype, I think it's a good trade-off.

Therefore, my evaluation dataset just needs to conform to the request & response API.
If we decide to use a completely different approach to measuring tag similarity, we can
do so in a way that is decoupled from the candidate source models and the ranking models.

```commandline
python cotag_evaluator.py \
    --input_data_path=data/cotag/test.jsonl \
    --output_data_path=data/cotag/gold.jsonl  # generates pairs of input and output tags
```
