import pandas as pd
import numpy as np
from warnings import filterwarnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

filterwarnings('ignore')
#loading the data set
anime = pd.read_csv('Anime.csv')
Ratings = anime[['Name','Type','Studio','Rating','Description']]

#Finding null values and replacing them
Ratings['Studio']=Ratings['Studio'].fillna('Unknown')

#Encoding the categorical variable Studio
encoder = dict([(j,i) for i,j in enumerate(Ratings['Studio'].value_counts().index)])
Ratings.set_index('Name',inplace=True)
Ratings['Studio'] = Ratings.apply(lambda row: encoder[row['Studio']],axis=1)

#Encoding the categorical variable Type
Type_encoder = dict([(j,i) for i,j in enumerate(Ratings['Type'].unique())])
Ratings['Type'] = Ratings.apply(lambda row: Type_encoder[row['Type']],axis=1)
Ratings.drop('Description',axis=1,inplace=True)


#Taking user input
anime_watched = input("What was the name of the latest anime you watched? ")

#finding closest matches to user input
def cosine_sim(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

if anime_watched not in Ratings.index:
    matches = anime['Name'].apply(lambda x: cosine_sim(anime_watched,x))
    matches.index = anime['Name']
    matches = matches.sort_values(ascending=False)
    matches = matches.to_frame()
    match_list = list(enumerate(matches.head().index))
    for i, j in match_list:
        print(f'{i}.{j}')

    match_list_dict = dict(match_list)
    choice = int(input(f'Seems like there are multiple animes with this title. Choose a number from 0 to 4 to confirm which anime you meant:'))
    anime_watched =match_list_dict[choice]

#Isolating the features on which recommendations should be given
Anime = Ratings[['Studio','Rating']]

#Finding similarities between user input title and existing database
Cos_Similarity = Anime.apply(lambda row: np.dot(Anime.loc[anime_watched],row)/(np.linalg.norm(Anime.loc[anime_watched])*np.linalg.norm(row)),axis=1)

#Adding the Studio column to get a better recommendation
Studio = Ratings['Studio']

#Converting to Dataframe
Cos = Cos_Similarity.to_frame()

#Adding columns to the dataframe
Cos.columns = ['Cosine Similarity']
Cos = Cos.join(Ratings['Studio'])

#Sorting recommendations by Cosine similarity
Recommendation = Cos.sort_values(by='Cosine Similarity',ascending=False)

n = int(input('Recommendations are ready. How many anime do want to be recommended (positive integer values only): '))

#Getting top 5 recommendations from the entire dataframe
recommended_n = Recommendation[Recommendation['Studio'] == Ratings.loc[anime_watched].loc['Studio']].head(n+1)

recommendation_list = list(recommended_n.index)
recommendation_list.remove(anime_watched)
print('\n\nRecommended to watch next:\n\n')
for i,j in enumerate(recommendation_list):
    print(f'{i+1}. {j}\n')