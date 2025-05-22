from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import string
import re
import emoji
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import rcParams
from textblob import TextBlob




extract = URLExtract()

import logging
logging.basicConfig(level=logging.DEBUG)

def fetch_stats(selected_user, df):
    logging.debug(f"Selected user: {selected_user}")
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the number of words
    words = []
    for msg in df['message']:
        words.extend(str(msg).split())

    # fetch the number of image messages
    num_image_messages = df[df['message'].str.contains('image omitted', case=False, na=False)].shape[0]

    # fetch the number of links shared
    links =[]
    for message in df['message']:
        links.extend(extract.find_urls(message))


    return  num_messages,len(words),num_image_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = df['user'].value_counts(normalize=True).mul(100).round(2).reset_index().rename(columns={'index': 'name', 'user': 'name'})

    return x, df

def create_wordcloud(selected_user, df):

    f = open("stop_hinglish.txt", encoding="utf-8")
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'message']
    temp = temp[temp['message'] != 'image omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)


    wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color = "white")
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=""))
    return df_wc

def most_common_words(selected_user,df):

    f = open("stop_hinglish.txt", encoding="utf-8")
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'message']
    temp = temp[temp['message'] != 'image omitted>\n']

    words = []
    remove_chars = string.punctuation + "“”‘’•–—"

    for message in temp['message']:
        message = re.sub(r'[^\x00-\x7F]+', '', message)  
        for word in message.lower().split():
            word = word.translate(str.maketrans('', '', remove_chars)).strip()
        if word and word not in stop_words and word not in ('omitted', 'image', 'document'):
            words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20),columns=['Word', 'Frequency'])
    return most_common_df

# emoji analysis
def plotly_emoji_pie_chart(emoji_df):
    top_emojis = emoji_df.head(10)
    fig = px.pie(
        top_emojis,
        values='Frequency',
        names='Emoji',
        title='Top 10 Emojis',
        hole=0.4
    )
    return fig

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []

    for message in df['message']:
        emojis.extend([char for char in message if char in emoji.EMOJI_DATA])
        
    emoji_counts = Counter(emojis)
    emoji_df = pd.DataFrame(emoji_counts.items(), columns=['Emoji', 'Frequency']).sort_values(by='Frequency', ascending=False)
    emoji_df.reset_index(drop=True, inplace=True)

    top_emojis = emoji_df.head(10)


    
    return emoji_df, top_emojis,

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    def get_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity

    df['sentiment'] = df['message'].apply(get_sentiment)

    # Classify into positive, neutral, negative
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

    sentiment_counts = df['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    return sentiment_counts, df
