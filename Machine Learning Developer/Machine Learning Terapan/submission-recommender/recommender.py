# %% [markdown]
# ## Nama : Alif Adwitiya Pratama

# %% [markdown]
# # Recommender System - Anime Recommendation

# %% [markdown]
# ### 1. Persiapan

# %% [markdown]
# #### 1.1 Masukkan Library

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
tf.test.gpu_device_name()

#remove warning
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# #### 1.2 Masukkan Data

# %%
dfAnime=pd.read_csv('./data/anime.csv')
#hanya menggunakan 100.000 data pertama
dfRating=pd.read_csv('./data/rating.csv',nrows=100000)

# %% [markdown]
# ### 2. Data Understanding

# %% [markdown]
# dataset yang digunakan dapat diakses [disini](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

# %% [markdown]
# #### 2.1 Tentang Dataset

# %% [markdown]
# **Deskripsi**<br>
# dataset ini berisi tentang data rating anime yang diambil dari [myanimelist.net](https://myanimelist.net/). Dataset ini berisi informasi tentang data preferensi pengguna dari 7.813.737 pengguna pada 12.294 anime yang berbeda yang dipisah kedalam dua dataset yaitu anime.csv yang berisi terkait informasi anime dan rating.csv yang berisi terkait rating user. Data ini diambil dari link kaggle [berikut.](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
# 
# **Tentang Fitur** <br>
# 
# Dataset **anime.csv** -> dapat digunakan untuk content based filtering
# 
# - anime_id - id unik yang mengidentifikasi sebuah anime
# - name - nama lengkap anime
# - genre - daftar genre anime yang dipisahkan oleh koma
# - type - tipe anime (film, TV, OVA, dll)
# - episodes - jumlah episode dalam sebuah anime (1 jika film)
# - rating - rata-rata rating dari 10 untuk anime tersebut
# - members - jumlah anggota komunitas yang ada di grup anime tersebut
# 
# Dataset **Rating.csv** -> dapat digunakan untuk collaborative filtering
# 
# - user_id - id pengguna yang tidak dapat diidentifikasi secara acak
# - anime_id - anime yang telah dinilai oleh pengguna tersebut
# - rating - rating dari 10 yang diberikan oleh pengguna tersebut (-1 jika pengguna menonton anime tersebut tapi tidak memberikan rating)
# 
# 
# 
# 

# %% [markdown]
# #### 2.2 Deskripsi Data
# 

# %% [markdown]
# ##### 2.2.1 Anime.csv

# %%
#sample data
dfAnime.head()

# %%
#shape data
print(dfAnime.shape)

# %%
# data type
dfAnime.info()

# episode perlu diperbaiki karena seharusnya integer
dfAnime['episodes']=dfAnime['episodes'].replace('Unknown',np.nan)
dfAnime['episodes']=dfAnime['episodes'].astype(float)

print('\n\nsesudah diperbaiki\n')

dfAnime.info()

# %%
# check null
display(dfAnime.isnull().sum())


# %%
# check duplicate
print('jumlah data duplikat : ',dfAnime.duplicated().sum())

# %%
# check deskripsi statistik
dfAnime.describe(include='all')


# %% [markdown]
# dari tabel diatas terdapat informasi bahwa :
# - Terdapat 2 judul anime yang sama tapi memiliki id berbeda
# - Terdapat 12.292 judul anime yang unik
# - Genre anime terdiri dari 3.264 jenis kombinasi genre berbeda
# - Tipe anime terdiri dari 6 jenis yang berbeda
# - Range jumlah episode anime sangat bervariasi, mulai dari 1  hingga 1818 episode dengan rata-rata 12 episode
# - Rating rata-rata anime adalah 6,47 dari skala 1 hingga 10
# - Jumlah rata-rata anggota komunitasi yang ada di grup anime adalah 1.807
# 

# %%
# nama anime yang duplikat
dfAnime[dfAnime.name.duplicated(keep=False)].sort_values(by='name')

#karna berbeda tidak perlu di drop

# %% [markdown]
# ##### 2.2.2 Rating.csv

# %%
#sample data
dfRating.head()

# %%
#shape data
print(dfRating.shape)

# %%
# data type
dfRating.info()


# %%
# check null
print('jumlah data null :\n',dfRating.isnull().sum())


# %%
# check deskripsi statistik
dfRating.describe()


# %% [markdown]
# dari tabel diatas terdapat informasi bahwa rating yang diberikan oleh user rata-rata adalah 6.14 dari skala -1 hingga 10 dimana -1 pada dataset ini berarti user tersebut menonton anime tersebut tapi tidak memberikan rating nantinnya nilai -1 ini akan diubah jadi 0
# 

# %% [markdown]
# ### 3. Content Based Filtering

# %% [markdown]
# #### 3.1 Data Preparation

# %% [markdown]
# ##### 3.1.1 Hanya Mengambil Fitur yang Dibutuhkan

# %% [markdown]
# karena disini akan dilakukan content based filtering maka hanya akan diambil fitur yang dibutuhkan yaitu anime_id, name, genre 

# %%
feature=['anime_id','name','genre']
dfAnimeClean=dfAnime[feature]


# %% [markdown]
# ##### 3.1.2 Hilangkan Null pada Genre

# %%
# remove null
dfAnimeClean=dfAnimeClean.dropna()
print('jumlah data null :\n',dfAnimeClean.isnull().sum())


# %% [markdown]
# ##### 3.1.3 ekstrak tfidf dari genre

# %%

#ekstrak tfidf genre
tfidf=TfidfVectorizer()
tfidf_matrix=tfidf.fit_transform(dfAnimeClean['genre'])
print(tfidf_matrix.shape)


# %%
print('fitur name : ',tfidf.get_feature_names_out())

# %%
pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tfidf.get_feature_names_out(),
    index=dfAnimeClean.name
).sample(10, axis=1).sample(10, axis=0)

# %% [markdown]
# ##### 3.1.4dapatkan cosine similarity dari tfidf

# %%
cosine_sim = cosine_similarity(tfidf_matrix) 
print(cosine_sim.shape)

# %%
cosine_sim_df=pd.DataFrame(cosine_sim, index=dfAnimeClean['name'], columns=dfAnimeClean['name'])
cosine_sim_df.sample(10, axis=1).sample(10, axis=0)

# %% [markdown]
# #### 3.2 Get Recommendation

# %%
def make_recommendations(nama_anime, similarity_data=cosine_sim_df, items=dfAnimeClean[['name', 'genre']], k=5):
   
 
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
    index = similarity_data.loc[:,nama_anime].to_numpy().argpartition(
        range(-1, -k, -1))
    
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # Drop nama_anime agar nama anime yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(nama_anime, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)

# %%
sample_user=dfRating[dfRating['user_id']==20]
print('anime yang disukai user :')
for i in sample_user['anime_id']:
	print(dfAnime[dfAnime['anime_id']==i]['name'].values[0],'->',sample_user[sample_user['anime_id']==i]['rating'].values[0],'| Genre :',dfAnime[dfAnime['anime_id']==i]['genre'].values[0])

# %%
rating_maksimal=sample_user['rating'].max()
anime_id_maksimal_rating=sample_user[sample_user['rating']==rating_maksimal]['anime_id'].values[0]
anime_name_maksimal_rating=dfAnime[dfAnime['anime_id']==anime_id_maksimal_rating]['name'].values[0]

result=make_recommendations(anime_name_maksimal_rating, k=10)
result

# %% [markdown]
# #### 3.3 Evaluation

# %%
# hapus nilai item yang dijadikan acuan. Karena pasti hit
sample_user_new=sample_user[sample_user['anime_id']!=anime_id_maksimal_rating]

# %%
rekomendasi_hit=0

# melakukan looping pada genre rekomendasi kemudian membandingkannya dengan genre yang pernah ditonton
for rekomendasi_genre in result['genre']:
    rekomendasi_genre=rekomendasi_genre.split(',')
    rekomendasi_genre=set(rekomendasi_genre)
    for watched_id in sample_user_new['anime_id']:
        watched_genre=dfAnime[dfAnime['anime_id']==watched_id]['genre'].values[0]
        watched_genre=watched_genre.split(',')
        watched_genre=set(watched_genre)
        if rekomendasi_genre.intersection(watched_genre):
            rekomendasi_hit+=1
            break

print('precision : ',rekomendasi_hit/len(result)*100,'%')
	

# %% [markdown]
# ### 4. Colaborative Based Filtering

# %% [markdown]
# #### 4.1 Data Preparation

# %% [markdown]
# #### 4.1.1 Ubah nilai -1 menjadi 0

# %%
dfRating['rating']=dfRating['rating'].replace(-1,0)
dfRating.head()

# %% [markdown]
# #### 4.1.2 encode anime_id dan user_id

# %%
user_ids = dfRating['user_id'].unique().tolist()
 
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

# %%
anime_ids = dfRating['anime_id'].unique().tolist()
 
anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded_to_anime = {i: x for i, x in enumerate(anime_ids)}

# %%
dfRating['user_id'] = dfRating['user_id'].map(user_to_user_encoded)
 
dfRating['anime_id'] = dfRating['anime_id'].map(anime_to_anime_encoded)

# %% [markdown]
# #### 4.1.2 Bagi dataset

# %%
# Mengacak dataset
dfRating = dfRating.sample(frac=1, random_state=42)
dfRating

# %%
min_rating = min(dfRating['rating'])
max_rating = max(dfRating['rating'])

x = dfRating[['user_id', 'anime_id']].values
 
y = dfRating['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
 
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * dfRating.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

# %% [markdown]
# #### 4.2 Modelling

# %%
class RecommenderNet(tf.keras.Model):
  
  # Insialisasi fungsi
  def __init__(self, num_users, num_anime, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_anime = num_anime
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.anime_embedding = layers.Embedding( # layer embeddings anime
        num_anime,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.anime_bias = layers.Embedding(num_anime, 1) # layer embedding anime bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    anime_vector = self.anime_embedding(inputs[:, 1]) # memanggil layer embedding 3
    anime_bias = self.anime_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_anime = tf.tensordot(user_vector, anime_vector, 2) 
 
    x = dot_user_anime + user_bias + anime_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

# %%
num_users = len(dfRating.user_id)
num_anime = len(dfRating.anime_id)
model = RecommenderNet(num_users, num_anime, 50)
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# %%


history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 500,
    epochs = 50,
    validation_data = (x_val, y_val),
)



# %% [markdown]
# #### 4.3 Evaluasi

# %%
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('RMSE Model')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %% [markdown]
# #### 4.4 Get Recommendation

# %%
 # Mengambil sample user
user_id = sample_user['user_id'].values[0]
anime_watched_by_user = dfRating[dfRating.user_id == user_id]
 
# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html 
anime_not_watched_by_user = dfRating[~dfRating['anime_id'].isin(anime_watched_by_user.anime_id.values)]['anime_id'] 
anime_not_watched_by_user = list(
    set(anime_not_watched_by_user)
     .intersection(set(anime_to_anime_encoded.keys()))
)
 
anime_not_watched_by_user = [[anime_to_anime_encoded.get(x)] for x in anime_not_watched_by_user]
user_encoder = user_to_user_encoded.get(user_id)

user_anime_array = np.hstack(
    ([[user_encoder]] * len(anime_not_watched_by_user), anime_not_watched_by_user)
)

# %%
ratings = model.predict(user_anime_array).flatten()
 
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_anime_ids = [
    anime_encoded_to_anime.get(anime_not_watched_by_user[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_id))

print('Top 10 anime recommendation')
print('----' * 8)
 
recommended_anime = dfAnime[dfAnime['anime_id'].isin(recommended_anime_ids)]
for row in recommended_anime.itertuples():
    print(row.name, ':', row.genre)


