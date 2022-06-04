
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_recommenders as tfrs

from typing import Dict, Text
from tensorflow.keras.optimizers import Adagrad

movies_f = 'movies.csv'
ratings_f = 'out.csv' #Shuffled version of rating.csv

movies_d = {}

with open(movies_f, 'r') as f:
  rows = f.readlines()
  for row in rows[1:]:
    row = row.split(",")
    movies_d[int(row[0])] = row[1]



data = []
ctr = 0
with open(ratings_f, 'r') as f:
  rows = f.readlines()
  for row in rows[1:]:
    row = row.split(",")
    
    r = {}
    
    #out.csv
    r["movie_title"] = tf.convert_to_tensor(movies_d[int(row[2])])
    r["user_id"] = tf.convert_to_tensor(row[1])
    r["user_rating"] = tf.convert_to_tensor(float(row[3]))

    

    data.append(r)

    ctr+=1
    
print(len(data))



df = pd.DataFrame(data)

ds = tf.data.Dataset.from_tensor_slices(df.to_dict(orient="list"))

mapped_movies_dataframe = ds.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

list(mapped_movies_dataframe.as_numpy_iterator())[0:5]

data_mapped = mapped_movies_dataframe





train = data_mapped.take(80_000)
test = data_mapped.skip(80_000).take(20_836)

user_ids = data_mapped.batch(1_000_836).map(lambda x: x["user_id"])
movie_titles = data_mapped.batch(1_000_836).map(lambda x: x["movie_title"])

unique_users = np.unique(np.concatenate(list(user_ids)))
unique_movies = np.unique(np.concatenate(list(movie_titles)))

class RatingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_size = 32

    
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_users, mask_token=None),
      tf.keras.layers.Embedding(len(unique_users) + 1, embedding_size)
    ])

    
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movies, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movies) + 1, embedding_size)
    ])

    
    self.ratings = tf.keras.Sequential([
      
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    user_id, movie_title = inputs

    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))



class RecommenderModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.rating_model: tf.keras.Model = RatingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError(),
               tf.keras.metrics.MeanAbsoluteError()]
    )

  def call(self, features) -> tf.Tensor:
    return self.rating_model(
        (features["user_id"], features["movie_title"]))

  def compute_loss(self, features, training=False) -> tf.Tensor:
    labels = features.pop("user_rating")

    rating_predictions = self(features)

    
    return self.task(labels=labels, predictions=rating_predictions)



model = RecommenderModel()
model.compile(optimizer=Adagrad(learning_rate=0.05))

train_batched = train.batch(8192).cache()
test_batched = test.batch(4096).cache()

model.fit(train_batched, epochs=50)

model.evaluate(test_batched, return_dict=True)

