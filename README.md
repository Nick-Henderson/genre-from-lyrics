# Can musical genre be predicted from just the lyrics?

The goal of this project is to find out whether musical genre can be predicted from just the lyrics. Given the somewhat nebulous and subjective nature of musical genres, it is possible this is a doomed classification task. I will make the strongest model that I can, and along the way I will see what I can learn about the connection between genre and lyrics. 

All of the code necessary to follow along with this project is contained within the notebook **Genre_from_lyrics_notebook.ipynb** 

### Data Sources

A Bag-of-Words lyrics dataset was attained from http://millionsongdataset.com/musixmatch/ . This dataset contains the counts for the top 5000 words in 200,000+ songs. 

The specific file links are as follows

* Word Counts:
http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip
and
http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_test.txt.zip

* Artist and Track information:
http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_779k_matches.txt.zip

* Unstemmed words dictionary:
http://millionsongdataset.com/sites/default/files/mxm_reverse_mapping.txt


Genre data was attained from several sources

* Allmusic genre annotations: 
http://www.ifs.tuwien.ac.at/mir/msd/partitions/msd-MAGD-genreAssignment.cls

* Tagtraum majority/minority genre annotatins:
https://www.tagtraum.com/genres/msd_tagtraum_cd2.cls.zip

* Tagtrum unambiguous annotations (single genre):
https://www.tagtraum.com/genres/msd_tagtraum_cd2c.cls.zip


Full raw lyrics were scraped from https://genius.com/ 


### Load and process data

*data should be stored in the 'data' folder, but it is not provided in this repo. To follow along, download the data from the links above to the data folder in your repo*

After data is downloaded into the data folder (with the default file names from the data sources), data can be processed with the Process_data class. There are four class methods that must be completed in the right order to create the files necessary for this project.

*Note: These methods create and save several numpy and csv files in the data folder along the way. While these would ideally all be created and loaded into memory in one easy to run method, the memory requirements make that impractical. Files are saved along the way is so that I (and you!) can keep less of the data in memory at any one time*

**1) Process_data.load_data()**

**2) Process_data.make_tfidf()**

**3) Process_data.clean_counts()**

**4) Process_data.clean_tfidf()**

After these four methods have been run, separate TF-IDF and counts dataframes can be loaded with **Process_data.load_tfidf()** and **Process_data.load_counts()**. These dataframes contain the TF-IDF or counts for the top 1382 words (top 1500 minus stopwords). Column headings are reverse-mapped unstemmed words to make for cleaner data presentation, but the actual data refers to the word stems. There are 101,884 songs in this dataset - this is the number of songs from the lyrics dataset for which genre labels were available in the genres dataset.

### Scrape additional data

The TF-IDF and counts dataframes will be great for data exploration and most modeling (e.g. regression and random forest models), but they will not be useful for deep learning models like LSTM, which requires the actual lyrics in their original structure. The original data source (http://millionsongdataset.com/musixmatch/) does not contain the raw lyrics due to copyright concerns. This may seem backwards given that I already have clean vectorized lyrics, but I will scrape the raw lyrics for as many of these songs as possible.

Lyrics were scraped from Genius.com. See notebook (**Genre_from_lyrics_notebook.ipynb**) for detail on scraping the lyrics. Raw scraped lyrics are available in the data folder so you will not need to scrape the lyrics to follow along with processing the lyrics and running the model. 

I was able to successfully scrape the the raw lyrics for 70,273 of the 101,884 songs in the counts/tfidf dataframes. For consistency, all modeling and EDA was limited to these 70,273 songs.

Now we have 3 datasets to work with for our EDA and modeling. 

1) Word Counts - DataFrame with the counts of the top 1382 words

2) TF-IDF - DataFrame with TF-IDF score for the top 1382 words

3) Raw Lyrics - Dataframe with the lyrics as a string 

## Exploratory Data Analysis

First lets look at the most common genres in the data. 

![](figs/distribution.png)  
![](figs/genre_counts.png)            

We can see that there are far more rock songs than any other genre. We will need to account for this feature of the dataset during the modeling section. We can also see that New Age has very few songs. To avoid complications, I have dropped all New Age songs from the dataset.

### Which genres have the largest vocabularies?

![](figs/unique_words_per_genre.png)  ![](figs/words_per_song.png)

Rap has by far the largest average vocabulary per song - that is, the largest number of unique words per song. We can see that rap also has the most total words per song, but not by as large of a margin.


### Which genres are most repetitive?

![](figs/repeats_per_word.png)

### What words are most common in different genres?

There are a lot of generic words that are popular in many songs across every genre. If I use the word counts to generate word clouds, I suspect that I will get very similiar clouds for most genres. To correct for this somewhat, I will instead use the TF-IDF data to generate word clouds. This should find words that are more unique.

### Blues
![](figs/cloud_blues.png)

### Metal
![](figs/cloud_metal.png)

### Electronic
![](figs/cloud_electronic.png)

### Reggae
![](figs/cloud_reggae.png)

We can see that these four genres are fairly distinct. Metal looks almost exactly how I think most would have guessed, and reggae has words like 'dem' that are not represented elsewhere. The top words for punk and rap are also fairly unique, but are a little NSFW so they aren't on display here. Conversely, the top words for rock, folk, pop and country (not shown) are much less unique. Of course, machine learning ought to be much better than my naked eye at picking up differences between these genres. 


## Modeling

Based on our EDA, it is unclear what we should expect from the accuracy our model. It seems quite possible that some genres will be very hard to distinguish, while others like rap, metal, and reggae, might be predicted more accurately. To evaluate models, I will be using accuracy score. For a simple, fairly low stakes classification problem like this one, a simple accuracy score seems like the best way to capture what we care about - whether the model predicted the correct genre. Models will be evaluated with 6 fold cross validation using 'StratifiedKFold' to maintain class proportions in each fold. 

### Baseline models:

For our baseline model, we can start by randomly assigning a class with a probability determined by class distribution - that is, we will be much more likely to assign 'Rock' (the most common genre) than blues (the least common genre). 

#### Accuracy Score: 0.25

We have very unbalanced clases, so perhaps a more fair baseline model is to just select the majority class for every prediction. 

#### Accuracy Score: 0.47


### Random Forest:



#### Accuracy Score: 0.54

The random forest model shows only a 7% improvement over the baseline model. This may indicate that the model is not well optimized, but I would argue that it more likely indicates that this is an especially challenging classification task. Lets take a look at some more details for one of the folds:

![](figs/rf_class_report.png)  ![](figs/rf_feature_importance.png)

Notice that in most cases the model struggles much more with recall than precision, except for Rock (the majority class) where it has very high recall but worse precision. This suggests that the model more or less defaults to Rock when it cannot 'confidently' label something as another genre, so it ends up labeling too many songs as Rock, and too few as the other genres. This is not actually that bad of a result (in my opinion). Functionally, it is somewhat useful to have a default class, and to only classify something as say, Blues when it is very 'Bluesy', Country when it is very Country, and Metal when it is very 'Metally'. That said, before settling for this sub-optimal classification, lets see if we can get stronger performance with a deep learning LSTM model. 

### LSTM

LSTM models differ from the above models in that they take into account the sequence of words. That is, they account for long and short-term dependencies within the data. For natural language processing this helps to capture features that would otherwise be missed. For our purposes, this might make a better classifier. For example, the same words might be common in one order in country songs, and in an entirely different order in pop songs. 

All processing of the data in preparation for LSTM can be found in the LSTM section of the notebook (**Genre_from_lyrics_notebook.ipynb**)

#### Cross Validation Accuracy Score: 0.56

The LSTM model performs just slightly better than the Random forest model. Again, I suspect that this is simply a difficult classification task given the nature of music genres. 

How did the LSTM model evolve over the epochs of training?

![](figs/lstm_loss.png)  ![](figs/lstm_accuracy.png)

We can see that the training accuracy continues increasing past the fourth epoch, but the test accuracy has already leveled off. Loss shows the same pattern. It appears that it only takes a few epochs to start overfitting our data. I am only showing a few epochs here, but upon running the model for 10 epochs, this pattern continued. The train accuracy got much higher (near 80%), but the test accuracy never improved past 55-56%. This suggests that the LSTM very quickly overfits data. It also suggests that, as I suspected, lyrics/genres are not very generalizable. Fitting very well to training data does not help make accurate predictions on unseen data. 

With the LSTM model as our final model, we can predict the genres for our validation data.

## Final Accuracy Score: 0.56

![](figs/lstm_class_report.png)

Compared to the random forest, this model is somewhat more balanced between precision and recall. Where there random forest was too likely to label songs as rock (98% recall, 52% precision), this model is a little less biased towards the majority class. Whether or not this is truly *better* is somewhat subjective. 


## Conclusions

This project has illustrated that not all classification tasks are created equally. In some classification tasks the labels are objective truth, and there is a direct connection between the label and the content. In the case of lyrics and genre, there are two clear disconnections - 1) the labels are subjective. No matter how high quality the genre labels are, they are still ultimately chosen based on the opinion of someone, somewhere. 2) There is not necessarily a direct connection between the label and the content. A country song can be about trucks and whiskey, but a country song can also be about love, or loss, or politics. This makes for a classification task that is never likely to be very accurate. 

This project has also illustrated the trade off between precision and recall. The random forest model was more precise for most genres, meaning it was unlikely to label something as country, metal, rap, etc. unless it truly belonged to those genres. This lead to worse recall - many songs were, by default, lumped in with rock, when they really belonged to other categories. The LSTM model improved recall slightly, at the expense of worse precision for most genres. 

Finally, we can conclude that some genres are more lyrically unique than others. Both models were much more successful labeling genres like rap and metal than they were labeling genres like pop/electronic/folk/blues. 


## Future Directions

1. Make a more modern dataset: The million songs dataset is quite dated, havihg been released in 2011. I worked with it due the the availability of lyrics and genre labels. In the future, I would like to scrape more modern data. 

2. Forget genres: Instead of attempting to use supervised learning to correctly classify lyrics into the correct genre, it might be more interesting to use unsupervised learning (e.g. LDA) to create topic labels. For example, it would be great to have a tool that easily let me find a song about love/sadness/etc.. 

3. Nearest neighbors: similar to the point above, it would be interesting to build a recommender that will return songs that are lyrically similar to an input song. 

4. Use music *and* lyrics: Make a model that uses both the features in the music and the features in the lyrics to classify genres. Would the lyrics help at all? or would the music tell the whole story?