import pandas as pd
import numpy as np

def extract_feature(item, feature, subtype=None):
    if subtype:
        try:
            feature_result = item.find(feature, subtype).get('value')
        except:
            feature_result = np.nan
    else:
        try:
            feature_result = item.find(feature).get('value')
        except:
            feature_result = np.nan
    return feature_result

def extract_multiple_features(item, feature):
    lst = []
    for feature in item.findAll('link', {'type':f'boardgame{feature}'}):
        lst.append(feature.get('value'))
    if len(lst) == 0:
        lst = 'N/A'
    return lst

def print_csv(games, i):
    games_df = pd.DataFrame.from_dict(games, orient='index', columns=['name', 'type', 'year', 'designer', 
                                                                      'artist', 'publisher', 'min_players', 
                                                                      'max_players', 'play_time', 'min_age', 
                                                                      'num_ratings', 'avg_rating', 'bayes_avg',
                                                                      'weight', 'categories', 'mechanics',
                                                                      'families'])
    games_df.to_csv(f'scraped_stats/games_to_{i}')