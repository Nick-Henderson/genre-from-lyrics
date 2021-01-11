# Can musical genre be predicted from just the lyrics?

The goal of this project is to find out whether musical genre can be predicted from just the lyrics. 

### Data Sources

Bag-of-Words lyrics dataset was attained from http://millionsongdataset.com/musixmatch/

The specific file links are as follows

* Word Counts:
http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip
and
http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_test.txt.zip

* Artist and Track information:
http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_779k_matches.txt.zip

* Unstemmed words dictionary:
http://millionsongdataset.com/sites/default/files/mxm_reverse_mapping.txt


Genres data was attained from several sources

* Allmusic genre annotations: 
http://www.ifs.tuwien.ac.at/mir/msd/partitions/msd-MAGD-genreAssignment.cls

* Tagtraum majority/minority genre annotatins:
https://www.tagtraum.com/genres/msd_tagtraum_cd2.cls.zip

* Tagtrum unambiguous annotations (single genre):
https://www.tagtraum.com/genres/msd_tagtraum_cd2c.cls.zip


Full raw lyrics were scraped from https://genius.com/ 



*Note: The following sections involve creating and saving several numpy and csv files by running class methods one-by-one. While these would ideally be done in one easy to run function, the memory requirements make that impractical. Files are saved along the way is so that I (and you!) can keep less of the data in memory at any one time*


### Load and process data

*data should be stored in the 'data' folder, but it is not provided in this repo. To follow along, download the data from the links above to the data folder in your repo*

After data is downloaded into the data folder (with the default file names from the data sources), data can be processed with the Process_data class. There are four class methods that must be completed in the right order to create the files necessary for this project.

**1) Process_data.load_data()**

**2) Process_data.make_tfidf()**

**3) Process_data.clean_counts()**

**4) Process_data.clean_tfidf()**

After these four methods have been run, separate TF-IDF and counts dataframes can be loaded with **Process_data.load_tfidf()** and **Process_data.load_counts()**. These dataframes contain the TF-IDF or counts for the top 1382 words (top 1500 minus stopwords). Column headings are reverse-mapped unstemmed words to make for cleaner data presentation, but the actual data refers to the word stems. 
