import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator

def recommend(name, names, df, cosine_sim, n):
    '''
    Returns recommendations from content-based recommendation system.
    
    Parameters:
    name - Boardgame to be compared, should be a string.
    names - Array of boardgame names.
    df - Dataframe of statistics, used to gather names for recommendations.
    cosine_sim - matrix of cosine similarities.
    n - number of recommendations to be returned.
    
    Returns:
    The names of (n) games similar to (name).
    '''
    recommended_games = []
    idx = names[names == name].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_n_indices = list(score_series.iloc[1:n+1].index)
    for i in top_n_indices:
        recommended_games.append(list(df['name'])[i])
    return recommended_games

def create_wordcloud(text):
    '''
    Creates a wordcloud figure.
    
    Parameters:
    text - Word pool used to make wordcloud.
    
    Returns:
    Worldcloud image.
    '''
    fig = plt.figure(figsize=(12,6))
    wordcloud = WordCloud(max_font_size=50, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()