import pandas as pd
from score import precompute_scores_v2

idx = pd.IndexSlice

test = precompute_scores_v2(full_version=True, min_zero=True, normalise=False, game=True)

# I am curious about something
test_df = pd.DataFrame.from_dict(test, orient='index', columns=['score'])
test_df.index = pd.MultiIndex.from_tuples(
    tuples=test_df.index.values.tolist()
    #,names=['bid_tricks', 'trump', 'actual_tricks', 'vul', 'double']
)

test_df['score'].min(), test_df['score'].max()