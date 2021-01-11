import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

class Process_data:
    
    def load_data():
        # Load the musiXmatch lyrics dataset
        # Data was split into test and train. We'll do our own test train split later.
        # So for now lets just combine all of it
        file1 = open('data/mxm_dataset_test.txt', 'r') 
        Lines = file1.readlines() 
        file2 = open('data/mxm_dataset_train.txt','r')
        Lines2 = file2.readlines()

        # The first row is a list of words - we will need these later as column names
        words = Lines[17][1:-1]
        words = words.split(',')
        # Combine all lines into one object
        data = Lines[18:] + Lines2[18:]

        # Collect the song ids from the dataset
        ids = []
        for row in data:
            r = row.split(',')
            ids.append(r[0])

        # Data is stored in ~almost dictionary format. Each row(song) has index:value pairs for every 
        # word featured in that song. We want to convert this into a matrix for use in modeling later

        # Make empty matrix
        d = np.zeros((len(data),5000))

        # Fill in the values from the dataset
        for i, row in enumerate(data):
            # Split the row into a list of word:count pairs
            r = row.split(',')
            r = r[2:]
            for word in r:
                # make the pair into an actual tuple, and put the value into the appropriate location in the matrix
                tup = word.split(':')
                idx = int(tup[0])-1
                val = int(tup[1])
                d[i,idx] = val


        # Now load track info and genre to combine with this dataset

        #Make dataframe with song ids from lyrics dataset
        df = pd.DataFrame(np.zeros((len(ids),1)))
        df.columns=['song_id']
        df.song_id = ids

        # load track info
        info = pd.read_csv('data/mxm_779k_matches.txt',sep='<SEP>',engine='python',header=None,skiprows=range(0,18)).drop(columns=[3,4,5])
        info.columns=['song_id','artist','title']

        # join track info with track ids to get the info for just the songs in our lyrics dataset
        song_info = df.join(info.set_index('song_id'), on='song_id')

        # Now we have song_id, artist, and title for all of our songs

        # load genre labels from multiple sources and join to info dataset on song_id
        genre1 = pd.read_csv('data/msd-MAGD-genreAssignment.cls',sep='\t',header=None)
        genre1.columns=['song_id','simple_genre']
        genre2 = pd.read_csv('data/msd_tagtraum_cd2.cls',sep='\t',skiprows=range(0,7),
                            names=['song_id','majority_genre','minority_genre'])
        genre3 = pd.read_csv('data/msd_tagtraum_cd2c.cls',sep='\t',skiprows=range(0,7),
                            names=['song_id','clean_genre'])


        song_info = song_info.join(genre1.set_index('song_id'),on='song_id')
        song_info = song_info.join(genre2.set_index('song_id'), on='song_id')
        song_info = song_info.join(genre3.set_index('song_id'), on='song_id')

        song_info['genre'] = song_info.clean_genre

        # Replace International with World to match other labels
        mask = song_info.simple_genre == 'International'
        song_info.loc[mask,'simple_genre'] = 'World'

        # Fill in missing 'genre' labels with the 'simple_genre' labels
        # when the label is one of the 'clean_genre' labels
        genres = song_info.clean_genre.unique()[1:]
        for genre in genres:
            mask = (song_info['simple_genre'] == genre) & (song_info['clean_genre'].isna())
            song_info.loc[mask,'genre'] = song_info.loc[mask,'simple_genre']

        # make an info dataframe with just the relevant columns
        info = song_info[['song_id','artist','title','genre']]

        # Select only the rows from the counts matrix and info dataframe where genre exists
        final_info = info[info['genre'].notna()]
        d = d[info['genre'].notna()]

        # Save info and data
        np.save('data/counts_data.npy',d)
        np.save('data/counts_trimmed.npy',d[:,0:1500])
        final_info.to_csv('data/final_info.csv',index=False)
        np.save('data/words.npy',np.array(words))

    def make_tfidf():
        # Load numpy counts data (created with load_data method)
        full_data = np.load('data/counts_data.npy')

        # Calculated document frequency and inverse document frequency
        doc_f = (full_data > 0).sum(axis=0)/len(full_data)
        idf = np.log(1/doc_f)

        # Calculate term frequency inverse document frequency
        tfidf = full_data*idf

        # Save tfidf (top 5000 words) and a trimmed tfidf (top 1500 words) as .npy files
        np.save('data/tfidf.npy',tfidf)
        np.save('data/tfidf_trimmed.npy',tfidf[:,0:1500])

    def clean_counts():
        # Load Counts
        counts = np.load('data/counts_trimmed.npy')
        counts = pd.DataFrame(counts)

        # load words for column headings
        words = np.load('data/words.npy')
        # Load reverse mapping to get unstemmed words for nicer column headings
        txt_map = pd.read_csv('data/mxm_reverse_mapping.txt',sep='<SEP>',header=None,engine='python')
        # Replace words list with unstemmed versions
        unstemmed = words.copy()
        for i in range(len(txt_map)):
            word = txt_map.iloc[i,0]
            replacement = txt_map.iloc[i,1]
            unstemmed[unstemmed == word] = replacement

        # Make unstemmed words into column headings
        counts.columns = unstemmed[0:1500]

        # Load track/genre info
        info = pd.read_csv('data/final_info.csv')

        # Join info with counts
        counts = pd.concat([info,counts],axis=1)

        # Remove stopwords columns
        stop = stopwords.words('english')

        for word in stop:
            if word in counts.columns:
                counts.drop(columns=word,inplace=True)

        counts.to_csv('data/counts_df.csv',index=False)

    def clean_tfidf():
        # Load TF-IDF
        tfidf = np.load('data/tfidf_trimmed.npy')
        tfidf = pd.DataFrame(tfidf)

        # load words for column headings
        words = np.load('data/words.npy')
        # Load reverse mapping to get unstemmed words for nicer column headings
        txt_map = pd.read_csv('data/mxm_reverse_mapping.txt',sep='<SEP>',header=None,engine='python')
        # Replace words list with unstemmed versions
        unstemmed = words.copy()
        for i in range(len(txt_map)):
            word = txt_map.iloc[i,0]
            replacement = txt_map.iloc[i,1]
            unstemmed[unstemmed == word] = replacement

        # Make unstemmed words into column headings
        tfidf.columns = unstemmed[0:1500]

        # Load track/genre info
        info = pd.read_csv('data/final_info.csv')

        # Join info with tfidf
        tfidf = pd.concat([info,tfidf],axis=1)

        # Remove stopwords
        stop = stopwords.words('english')

        for word in stop:
            if word in tfidf.columns:
                tfidf.drop(columns=word,inplace=True)

        tfidf.to_csv('data/tfidf_df.csv',index=False)
            
    
    def load_tfidf():

        tf = pd.read_csv('data/tfidf_df.csv')

        return tf

    def load_counts():

        cts = pd.read_csv('data/counts_df.csv')

        return cts