# Make HTTP requests
import requests
# Scrape data from an HTML document
from bs4 import BeautifulSoup
# I/O
import os
# Search and manipulate strings
import re
import pandas as pd

def scrape_song_lyrics(url):
    try:
        page = requests.get(url)
        html = BeautifulSoup(page.text, 'html.parser')
        lyrics = html.find('div', class_='lyrics').get_text()
        #remove identifiers like chorus, verse, etc
        lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
        #remove empty lines
        lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
        lyrics = lyrics.replace('\n', ' ')
        return lyrics
    except:
        return 'not found'

info = pd.read_csv('~/Desktop/genre-from-lyrics/data/final_info.csv')
info['url'] = 'https://genius.com/' + (info.artist.str.replace(' ','-').str.capitalize().str.replace("'", "") + 
 '-' + info.title.str.replace(' ','-').str.lower().str.replace("'", "") + '-lyrics').str.replace('.','').str.replace('&','and')

short = info.iloc[80000:90000]
short['lyrics'] = short.url.apply(scrape_song_lyrics)
short.to_csv('~/Desktop/genre-from-lyrics/data/lyrics9.csv')