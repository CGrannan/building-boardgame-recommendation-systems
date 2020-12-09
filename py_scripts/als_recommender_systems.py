from pyspark.ml.recommendation import ALS 

def new_user_recs(user_id, new_ratings, rating_df, stats_df, num_recs, spark):
  '''
  Returns a set number of recommendations for a new user.
  
  Parameters:
  user_id - New user's id number, must be an integer.
  new_ratings - New user's ratings, the more, the better.
  rating_df - Dataframe with all user's ratings.
  stats_df - dataframe containing names and game_ids
  num-recs - Number of recommendations to be returned.
  spark - Spark session
  
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

def create_new_recommendations(rating_df, stats_df, num_recs, spark):
  '''
  Prompts user to input names and ratings for games, then returns recommendations
  
  Parameters:
  rating_df - Dataframe with all user's ratings.
  stats_df - dataframe containing names and game_ids
  num-recs - Number of recommendations to be returned.
  spark - Spark session
  
  Returns:
  Top (num_recs) games recommended.
  '''
  user = 1000000
  cont = True
  ratings = []
  while cont == True:
    game = input('Enter a game for recommendations. ')
    rating = int(input('Enter rating. '))
    ratings.append((game, rating, user))
    y_n = input('Rate more games? y/n ').lower()
    if y_n == 'n':
      cont = False
  new_user_recs(user_id= user,
                new_ratings= ratings,
                rating_df= rating_df,
                stats_df= stats_df, 
                num_recs= num_recs,
                spark=spark)