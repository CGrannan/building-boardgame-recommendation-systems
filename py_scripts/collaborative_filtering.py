from sklearn.decomposition import TruncatedSVD
import numpy as np

def get_knn_recommendations(game_name, model, stats_df, pvt_table, pvt_table2):
    '''
    Returns recommendations from KNN recommendation system.
    
    Parameters:
    game-name - name of boardgame to be compared.
    model - NearestNeighbors model used to make recommendations.
    stats_df - Statistics dataframe, used to pull names.
    pvt_table - Matrix with game IDs, used to find indices. 
    pvt_table2 - Matrix of ratings used to find similarity.
    
    Returns:
    Names of 5 games most similarly rated to (game_name).
    '''
    game_id = stats_df[stats_df['name'] == game_name].game_id.item()

    game_index = pvt_table[pvt_table.game_id == game_id].index
    distances, indices = model.kneighbors(pvt_table2.iloc[game_index, :].values.reshape(1, -1), n_neighbors = 6)

    names = []
    for index in indices[0]:
        rec_idx = pvt_table2.index[index]
        names.append(stats_df[stats_df['game_id'] == rec_idx].name.item())
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(game_name))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, names[i], distances.flatten()[i]))
            
def get_svd_recommendations(game_name, n_components=20, stats_df, pvt_table, pvt_table2):
    '''
    Returns recommendations from SVD recommendation system.
    
    Parameters:
    game-name - name of boardgame to be compared.
    n_components - How many components decomposed matrix will contain.
    stats_df - Statistics dataframe, used to pull names.
    pvt_table - Matrix of ratings used to find similarity.
    pvt_table2 - Matrix with game IDs, used to find indices. 
    
    Returns:
    Names of 5 games most similarly rated to (game_name).
    '''
    svd = TruncatedSVD(n_components=n_components)
    matrix = svd.fit_transform(pvt_table)
    corr = np.corrcoef(matrix)
    ids = pvt_table.index
    
    game_id = stats_df[stats_df['name'] == game_name].game_id.item()
    idx = pvt_table2[pvt_table2.game_id == game_id].index[0]
    recs = list(ids[corr[idx] > .9])
    for game in recs:
        name = stats_df[stats_df['game_id'] == game].name.item()
        print(name)