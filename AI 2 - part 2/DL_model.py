#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pickle


# In[2]:


df= pd.read_csv('vgsales.csv')

print(df.head)


# In[3]:


print(df.isnull().sum())


# In[4]:


df.dropna(inplace=True)

df['Year'] = pd.to_datetime(df['Year'], format='%Y')


# In[5]:


sales_data = df[['Global_Sales', 'NA_Sales']]


# In[6]:


def categorize_continent(row):
    if row['NA_Sales'] > 0:
        return 'North America'
    elif row['EU_Sales'] > 0:
        return 'Europe'
    elif row['JP_Sales'] > 0:
        return 'Japan'
    else:
        return 'Other'

df['Continent'] = df.apply(categorize_continent, axis=1)


# In[7]:


publisher_sales = df.groupby('Publisher')['Global_Sales'].sum().reset_index()


# In[8]:


genre_sales = df.groupby('Genre')['Global_Sales'].sum().reset_index()


# In[9]:


genre_sales_sorted = genre_sales.sort_values(by='Global_Sales', ascending=False)


# In[10]:


publisher_sales_sorted = publisher_sales.sort_values(by='Global_Sales', ascending=False)


# In[11]:


top_n_publishers = 15 #This number can be changed to get different results in the chart


# In[12]:


top_n_genres = 15 #This number can be changed to get different results in the chart


# In[13]:


continent_sales = df.groupby(['Publisher', 'Continent'])['Global_Sales'].sum().reset_index()


# In[14]:


largest_publishers_by_continent = continent_sales.groupby('Continent').apply(lambda x: x['Publisher'][x['Global_Sales'].idxmax()]).reset_index()


# In[15]:


plt.figure(figsize=(12, 6))
plt.bar(publisher_sales_sorted['Publisher'][:top_n_publishers], publisher_sales_sorted['Global_Sales'][:top_n_publishers])
plt.xticks(rotation=90)
plt.xlabel('Publisher')
plt.ylabel('Global Sales')
plt.title('Top {} Publishers by Global Sales'.format(top_n_publishers))


# In[16]:


plt.savefig('top_publishers_global_sales.png')
plt.show()


# In[17]:


plt.figure(figsize=(12, 6))
plt.bar(genre_sales_sorted['Genre'][:top_n_genres], genre_sales_sorted['Global_Sales'][:top_n_genres])
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Global Sales')
plt.title('Top {} Genres by Global Sales'.format(top_n_genres))


# In[18]:


plt.savefig('top_genres_global_sales.png')
plt.show()


# In[19]:


print("Largest Overall Publisher:")
print(largest_publisher)


# In[20]:


print("Largest Publishers by Continent:")
print(largest_publishers_by_continent)


# In[21]:


num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
df['Cluster'] = kmeans.fit_predict(sales_data)


# In[22]:


plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(data=df, x='Global_Sales', y='NA_Sales', hue='Cluster', palette='dark', legend='full')
plt.xlabel('Global Sales')
plt.ylabel('NA Sales')
plt.title('Clustering of Games by Sales')


# In[ ]:


#DL


# In[42]:


df = pd.read_csv('vgsales.csv')


# In[43]:


df.dropna(inplace=True)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')


# In[56]:


df['Year'] = df['Year'].dt.year


# In[57]:


X = df[['Year', 'Genre', 'Platform']]
y = df['Global_Sales']


# In[58]:


encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = encoder.fit_transform(X[['Genre', 'Platform']])


# In[59]:


encoded_feature_names = encoder.get_feature_names_out(['Genre', 'Platform'])


# In[60]:


X_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names)
X = pd.concat([X[['Year']], X_encoded], axis=1)


# In[61]:


print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# In[62]:


min_samples = min(X.shape[0], y.shape[0])
X = X.iloc[:min_samples]
y = y.iloc[:min_samples]


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[64]:


model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1) 
])


# In[65]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[66]:


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# In[69]:


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[108]:


def preprocess_new_data(new_data, encoder, encoded_feature_names):
    # Data preprocessing (similar to training data)
    new_data['Year'] = pd.to_datetime(new_data['Year'], format='%Y')
    new_data['Year'] = new_data['Year'].dt.year

    # Select relevant features for prediction
    new_data = new_data[['Year', 'Genre', 'Platform']]

    # One-hot encoding for categorical variables (Genre and Platform)
    new_data_encoded = encoder.transform(new_data[['Genre', 'Platform']])
    new_data_encoded = pd.DataFrame(new_data_encoded, columns=encoded_feature_names)
    new_data = pd.concat([new_data[['Year']], new_data_encoded], axis=1)

    return new_data


# In[109]:


new_game = pd.DataFrame({
    'Year': [2023],       # Replace with the desired values
    'Genre': ['Action'],
    'Platform': ['PS5']
})


# In[110]:


with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)
encoded_feature_names = encoder.get_feature_names_out(['Genre', 'Platform'])


# In[111]:


new_game_features_encoded = preprocess_new_data(new_game, encoder, encoded_feature_names)


# In[112]:


predicted_sales = model.predict(new_game_features_encoded)


# In[106]:


print("Predicted Global Sales:", predicted_sales[0][0])


# In[ ]:




