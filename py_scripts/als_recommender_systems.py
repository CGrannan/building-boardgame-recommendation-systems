from pyspark.ml.recommendation import ALS 

def new_user_recs(user_id, new_ratings, rating_df, stats_df, num_recs, spark):
  '''
  Returns a set number of recommendations for a new user.
  
  Parameters:
  user_id - New user's id number, must be an integer.
  new_ratings - New user's ratings, the more, the better.
  rating_df - Dataframe with all user's ratings.
  num-recs - Number of recommendations to be returned.
  
  Returns:
  A list of game names and their predicted ratings for the new user.
  '''
  for i, rating in enumerate(new_ratings):
    game_id = stats_df[stats_df['name'] == rating[0]].game_id.item()
    new_ratings[i] = (int(game_id), rating[1], rating[2])

  new_ratings_df = spark.createDataFrame(new_ratings, schema=rating_df.columns)
  new_df = rating_df.union(new_ratings_df)

  als = ALS(rank=4, regParam=0.005, userCol="user_id", itemCol="game_id", ratingCol="rating", coldStartStrategy="drop")
  model = als.fit(new_df)

  recommendations = model.recommendForAllUsers(num_recs)
  recs = recommendations.where(recommendations.user_id == user_id).take(1)

  for rank, (game, rating) in enumerate(recs[0].recommendations):
    name = stats_df[stats_df['game_id'] == game]['name'].item()
    print('Recommendation {}: {} | Predicted Score = {}'.format(rank+1, name, round(rating, 2)))