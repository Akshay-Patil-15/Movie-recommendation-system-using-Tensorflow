import pandas as pd
import math
import tensorflow as tf
import numpy as np
import tensorflow_recommenders as tfrs
from typing import Dict, Text
from tensorflow.keras.optimizers import Adagrad


ratings_f = 'out.csv'             #Shuffled version of rating.csv
df_out = pd.read_csv("out.csv")



data = []
ctr = 0
with open(ratings_f, 'r') as f:
  rows = f.readlines()
  for row in rows[1:]:
    row = row.split(",")
    
    r = {}
    
    
    r["movie_id"] = tf.convert_to_tensor(row[2])
    r["user_id"] = tf.convert_to_tensor(row[1])
    r["user_rating"] = tf.convert_to_tensor(float(row[3]))

    

    data.append(r)

    ctr+=1


df = pd.DataFrame(data)

ds = tf.data.Dataset.from_tensor_slices(df.to_dict(orient="list"))

data_mapped = ds.map(lambda x: {
    "movie_id": x["movie_id"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})






train = data_mapped.take(80_000)
test = data_mapped.skip(80_000).take(20_836)

movie_titles = data_mapped.batch(1_000_836).map(lambda x: x["movie_id"])
user_ids = data_mapped.batch(1_000_836).map(lambda x: x["user_id"])

unique_movies = np.unique(np.concatenate(list(movie_titles)))
unique_users = np.unique(np.concatenate(list(user_ids)))

class RankingModel(tf.keras.Model):

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
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError(),
               tf.keras.metrics.MeanAbsoluteError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["user_id"], features["movie_id"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")

    rating_predictions = self(features)

    
    return self.task(labels=labels, predictions=rating_predictions)



model = RecommenderModel()
model.compile(optimizer=Adagrad(learning_rate=0.1))

train_batched = train.batch(8192).cache()
test_batched = test.batch(4096).cache()

model.fit(train_batched, epochs=50)

model.evaluate(test_batched, return_dict=True)

train = df_out[:80000]
test = df_out[80000:]

y_train = {}
for index, row in train.iterrows():
  y_train[(int(row['userId']),int(row['movieId']))] =  1

# Get Top 10 recommendation for each user.
y_pred = {}
y_true = {}
c = 0
for index, row in test.iterrows():
  y_true[row['userId']] =  y_true.get(row['userId'],[]) + [row['movieId']]
  
  if row['userId'] not in y_pred:
    c+=1
    
    for i in range(len(unique_movies)):
      
      if (int(row['userId']),int(unique_movies[i])) not in y_train:
        
        temp_tensor = model({
          "user_id": np.array([str(int(row['userId']))]),
          "movie_id": np.array([str(int(unique_movies[i]))])
          })
        val = temp_tensor.numpy()[0][0]
        y_pred[row['userId']] =  y_pred.get(row['userId'],[]) + [(int(unique_movies[i]),val)]

for k,v in y_pred.items():
  y_pred[k].sort(key=lambda x:x[1],reverse=True)

def recall(y_true, y_pred):
  ctr= 0
  
  x = y_pred[:10]
  y_list = y_true
  x_list = []
  for i in range(len(x)):
    x_list.append(x[i][0])
    
  

  for i in x_list:
    if(i in y_list):
      ctr+=1
  return ctr/len(x)

def precision(y_true, y_pred):
  ctr= 0

  x = y_pred[:10]
  y_list = y_true
  x_list = []

  for i in range(len(x)):
    x_list.append(x[i][0])
    
  
  for i in x_list:
    if(i in y_list):
      ctr+=1
  return ctr/len(y_list)

def f1_score(precision,recall):
  return 2*precision*recall/(precision+recall)

def recall_score(y_true, y_pred):
  accumulative_recall = 0
  for i in range(1,610):
    accumulative_recall+=recall(y_true[i], y_pred[i])
  return accumulative_recall/610 

print("Recall: {:.4f}".format(recall_score(y_true, y_pred)))

def precision_score(y_true, y_pred):
  accumulative_precision = 0
  for i in range(1,610):
    accumulative_precision+=precision(y_true[i], y_pred[i])
  return accumulative_precision/610

print("Precision: {:.4f}".format(precision_score(y_true, y_pred)))

def f1_score(y_true,y_pred):
  p = precision_score(y_true, y_pred)
  r = recall_score(y_true, y_pred)
  return 2*p*r/(p+r)

print("F1-Score: {:.4f}".format(f1_score(y_true, y_pred)))

def nDCG(y_true, y_pred):

  ctr= 2
  
  x = y_pred[:10]
  y_list = y_true
  x_list = []
  for i in range(len(x)):
    x_list.append(x[i][0])
    
  
  dcg = 0
  idcg = 0
  for i in range(len(x_list)):
    if(x_list[i] in y_list):
      dcg += 1/(math.log(i+2,2))
      idcg += 1/(math.log(ctr,2))
      ctr+=1

  if idcg == 0:
    return 0
  return dcg/idcg

def nDCG_score(y_true, y_pred): 
  accumulative_nDCG = 0
  for i in range(1,610):
    if(len(y_true[i])==1):
      accumulative_nDCG+=1
    else:
      accumulative_nDCG+=nDCG(y_true[i], y_pred[i])
  return accumulative_nDCG/610

print("nDCG-Score: {:.4f}".format(nDCG_score(y_true, y_pred)))

