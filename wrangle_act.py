#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries and getting data into jupyter

# In[4]:


#importting libraries
import requests
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pls


# In[2]:


#TSV url
url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'


# In[3]:


# request to tsv url and store data into file
response = requests.get(url)
with open (url.split('/')[-1], mode='wb') as file :
    file.write(response.content)


# In[4]:


#checking file list
os.listdir()


# In[5]:


# make list of tweets id
twitter_archive_enh = pd.read_csv('twitter-archive-enhanced.csv')
tweet_ids = list(twitter_archive_enh.tweet_id)
len(tweet_ids)


# In[5]:


import tweepy
from timeit import default_timer as timer
import json


# In[ ]:


consumer_key = 'bvzJAYUMWhqQTmZid8yqnVXYA'
consumer_secret = 'jTrHT7bDVhAShmaW1Z3BRXHrVj3fixyYG4FG0EpgUirWBqKPCk'
access_token = '372271092-vHKFE2yNEf8JM3nj8RftPq3kREFzckuOXtq74gIF'
access_secret = 'yZpupERQaYD2f6If0b1obi3l8hf03eK3VZaBS3uMeygii'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

count = 0
fails_dict = {}
start = timer()
# Save each tweet's returned JSON as a new line in a .txt file
with open('tweet_json.txt', 'w') as outfile:
    # This loop will likely take 20-30 minutes to run because of Twitter's rate limit
    for tweet_id in tweet_ids:
        count += 1
        print(str(count) + ": " + str(tweet_id))
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            print("Success")
            json.dump(tweet._json, outfile)
            outfile.write('\n')
        except tweepy.TweepError as e:
            print("Fail")
            fails_dict[tweet_id] = e
            pass
end = timer()
print(end - start)
print(fails_dict)


# In[7]:


#checking file list
os.listdir()


# ## See overview of all data

# In[6]:


# importing data to pandas dataframe
twitter_archive_enh = pd.read_csv('twitter-archive-enhanced.csv')
image_predictions = pd.read_csv('image-predictions.tsv', sep='\t')


# In[7]:


#open json data
json_data = [json.loads(line) for line in open('tweet_json.txt', 'r')]


# In[8]:


#get list of keys in dictionary
list(json_data[0].keys())


# In[9]:


json_data[0]['full_text']


# In[10]:


# take several parameter from dictionary data into list
json_conv = []

for datum in json_data:
    json_conv.append([datum['id'],datum['full_text'], datum['in_reply_to_status_id'], datum['in_reply_to_user_id'], datum['retweet_count'], datum['favorite_count']])


# In[11]:


#change list into df
json_df = pd.DataFrame(json_conv, columns=['tweet_id', 'full_text', 'in_reply_to_status_id', 'in_reply_to_user_id', 'retweet_count','favorite_count'])


# ### twitter_archive_enhanced

# In[12]:


# see head and tail of twitter archive enhanced
twitter_archive_enh


# - find wrong column name, floof instead of floofer
# - timestamp has date and time in one column

# In[13]:


twitter_archive_enh.info() 


# timestamp and retweeted_status_timestamp must be in date type
# in_reply_to_status_id and in_reply_to_user_id must be in int type

# In[14]:


twitter_archive_enh[twitter_archive_enh['retweeted_status_id'].notnull()]


# several Retweet data in this dataframe

# In[15]:


twitter_archive_enh.describe()


# Strange value in rating_numerator and rating_denominator since number of max is very far from mean data and 75% data

# In[16]:


# adding new column will help the investigation
# calculate normalize rating by dividing rating_numerator and rating/denominator
twitter_archive_enh['normalize_rating'] = twitter_archive_enh['rating_numerator']/twitter_archive_enh['rating_denominator']


# In[20]:


twitter_archive_enh[['rating_numerator', 'rating_denominator', 'normalize_rating']].describe()


# there are inf value in normalize rating, which is really strange value since median is 1.10. It is probably occure because of 0 value in rating denominator. Deep exploration needed in this value.

# In[22]:


twitter_archive_enh.normalize_rating.sort_values(ascending=False)


# In[23]:


# see data when in reply to status is not null
twitter_archive_enh[twitter_archive_enh.in_reply_to_status_id.notnull()]


# In[25]:


twitter_archive_enh.query('normalize_rating==inf')


# rating_numerator is very big but rating_denominator is 0. let's check the value in text.

# In[29]:


# get index of infinite number
twitter_archive_enh.query('normalize_rating==inf').index


# In[27]:


list(twitter_archive_enh.query('normalize_rating==inf').text)


# in fact, value stored in rating_numerator and rating_denominator is wrong. It it 13/10 instead of 960/0. So rating_numerator and rating denominator in row 313 need to be fixed into 13/10

# In[ ]:


twitter_archive_enh['text'][263]


# In[26]:


twitter_archive_enh['text'][571]


# text column including content and link

# ### image_prediction

# In[30]:


image_predictions.head(20)


# In[31]:


image_predictions.info()


# In[32]:


image_predictions.describe()


# In[33]:


image_predictions.jpg_url.value_counts()


# several jgp_url has duplication, it needs further inspection to make sure there are a duplication of image or not

# In[34]:


image_predictions[image_predictions['jpg_url']=='https://pbs.twimg.com/media/CsGnz64WYAEIDHJ.jpg']


# there are duplication data with different tweet_id

# In[35]:


dup = image_predictions[image_predictions.duplicated(subset=['jpg_url', 'img_num'], keep =False)]
dup.sort_values('jpg_url')


# dog prediction will be more clear without '_' and '-' separator and all character must re-cast into lowercase

# ### Json_dataframe

# In[36]:


json_df


# In[37]:


json_df.info()


# In[38]:


json_df.describe()


# max value in retweet_count and favourite_count is unusual, deeper inspection is needed

# In[39]:


# adding favorite_retweet_ratio that calculated from favorite_count/retweet_count
json_df['favorite_retweet_ration'] =  json_df['favorite_count']/json_df['retweet_count']


# In[40]:


json_df.describe()


# In[41]:


json_df.query('(retweet_count>3156) & (favorite_count<3307)')


# It is weird to see favorite_retweet_ratio is 0, in fact that tweet has big retweet count. 
# With average favorite_retweet_ratio around 3.22, favorite count should be bigger than retweet count

# In[42]:


json_df.query('favorite_retweet_ration>4.07').sort_values('favorite_retweet_ration', ascending=False)


# In[43]:


cek_rt_json = json_df.full_text.str.extract(r'(RT @.*)')
cek_rt_json[cek_rt_json[0].notnull()]


# found Retweeted tweets in json dataframe

# ## Assess data note

# ## Quality

# #### twitter atchive enhanced

# - timestamp and retweeted status timestamp has type string, which will good in date type

# - wrong column name, floofer must be floof

# - in_reply_to_status_id and in_reply_to_user_id type is in float, but it must be in int

# - Retweet data is in dataframe must be deleted

# - rating_numerator and rating_denominator in row 313 value on normalize_rating is 13/10 instead of 960/0

# ### image prediction

# - dog prediction will be more clear by replacing '_' separator into space

# - Several data duplication stored with different tweet_id

# - name of all dogs predictions are variate with upper and lower case

# ### Json data

# - retweeted tweets is in data

# ## Tidy

# ### twitter archive enhanced

# - timestamp has date and time in one column

# - text column value has two informations, there are tweet content and link

# ## Cleaning data process in quality

# ### copy dataframe

# In[73]:


twit_arc_copy = twitter_archive_enh.copy()


# In[74]:


img_pred_copy = image_predictions.copy()


# In[75]:


json_copy = json_df.copy()


# ### `timestamp and retweet_timestamp has string type, which must be date`

# ### Define

# timestamp and retweet_timestamp column type will change into date type

# ### Code

# In[76]:


#inspect value in timestamp
twit_arc_copy.timestamp.head()


# In[77]:


#change data type in timestamp into datetime64
twit_arc_copy['timestamp'] = twit_arc_copy.timestamp.astype('datetime64')


# In[78]:


#change data type in retweet_status_timestamp into datetime64
twit_arc_copy['retweeted_status_timestamp'] = twit_arc_copy.retweeted_status_timestamp.astype('datetime64')


# In[79]:


twit_arc_copy[['timestamp', 'retweeted_status_timestamp']]


# ### Test

# In[80]:


# check data type of changed column 
twit_arc_copy.info()


# ### `wrong column name, floof insted of floofer`

# ### Define

# rename floofer column into floof

# ### Code

# In[81]:


# Change column named 'floofer' become 'floof'
twit_arc_copy.rename(columns={'floofer':'floof'}, inplace = True)


# In[82]:


twit_arc_copy.head()


# ### Test

# In[83]:


# listing all column names in twit archive clean
list (twit_arc_copy.columns)


# ### `in_reply_to_status_id and in_reply_to_user_id should be in int, present in float`

# ### Define

# Change in_reply_to_status_id type into int, either do in_reply_to_user_id type

# ### Code

# In[84]:


twitter_archive_enh.tail()


# In[85]:


# change null value into 0
twit_arc_copy.in_reply_to_status_id = twit_arc_copy.in_reply_to_status_id.fillna(0)
twit_arc_copy.in_reply_to_user_id = twit_arc_copy.in_reply_to_user_id.fillna(0)


# In[86]:


twit_arc_copy[twit_arc_copy.in_reply_to_status_id.isnull()]


# In[87]:


# change in_reply_to_status_id and in_reply_to_user_id data type to int
twit_arc_copy.in_reply_to_status_id = twit_arc_copy.in_reply_to_status_id.astype(int)
twit_arc_copy.in_reply_to_user_id = twit_arc_copy.in_reply_to_user_id.astype(int)


# In[88]:


twit_arc_copy.info()


# In[89]:


twit_arc_copy.tail()


# ### Test

# In[90]:


twit_arc_copy['in_reply_to_user_id'].unique()


# In[91]:


twit_arc_copy.info()


# In[92]:


twit_arc_copy[twit_arc_copy['in_reply_to_status_id'].notnull()]


# ### `rating_numerator and rating_denominator in row 313 is 13/10 instead of 960/0`

# ### Define

# - change rating_numerator and rating_denominator in row 313 change into 13 and 10

# ### Code

# In[105]:


# find row of inf normalized_rating
twit_arc_copy.query('normalize_rating==inf').index


# In[106]:


twit_arc_copy.rating_numerator[313] = 13
twit_arc_copy.rating_denominator[313] = 10


# In[107]:


twit_arc_copy.normalize_rating[313] = twit_arc_copy.rating_numerator[313]/twit_arc_copy.rating_denominator[313]


# In[108]:


# recheck denominator condition
twit_arc_copy[twit_arc_copy['rating_denominator']==0]


# ### Test

# In[109]:


twit_arc_copy.describe()


# ### `Retweet data in dataframe must be deleted`

# ### Define

# - find retweet data in dataframe, get index and delete it

# ### Code

# In[110]:


#find Retweet tweet id
rt_tweet = twit_arc_copy[twit_arc_copy.retweeted_status_id.notnull()]['tweet_id'].values.tolist()
len(rt_tweet)


# In[111]:


test_cp = twit_arc_copy.copy()


# In[112]:


#delete rows based on tweet id
for x in rt_tweet:
    rt_ind = twit_arc_copy[twit_arc_copy['tweet_id']== x].index
    twit_arc_copy.drop(rt_ind, inplace=True)


# ### Test

# In[113]:


twit_arc_copy[twit_arc_copy.retweeted_status_id.notnull()]


# ### `dog prediction will be more clear by replacing '_' with space` 

# ### Define

# - replace ' _ ' with ' ' in column p_1, p_2 and p_3

# ### Code

# In[114]:


# col name checking
list(img_pred_copy.columns)


# In[115]:


# change '_' and '-' with space
img_pred_copy['p1'] = img_pred_copy['p1'].str.replace('_', ' ', regex=True) 
img_pred_copy['p2'] = img_pred_copy['p2'].str.replace('_', ' ', regex=True) 
img_pred_copy['p3'] = img_pred_copy['p3'].str.replace('_', ' ', regex=True)
img_pred_copy['p1'] = img_pred_copy['p1'].str.replace('-', ' ', regex=True) 
img_pred_copy['p2'] = img_pred_copy['p2'].str.replace('-', ' ', regex=True) 
img_pred_copy['p3'] = img_pred_copy['p3'].str.replace('-', ' ', regex=True)


# ### Test

# In[116]:


img_pred_copy


# ### `duplication data stored with different tweet_id` 

# ### Define

# - find duplicate data and drop it from dataframe

# ### Code

# In[117]:


#delete rows based on tweet id
for x in rt_tweet:
    rt_ind = img_pred_copy[img_pred_copy['tweet_id']== x].index
    img_pred_copy.drop(rt_ind, inplace=True)


# ### Test

# In[118]:


# check if it still duplication jpg_url or retweeted tweet in dataframe
img_pred_copy[img_pred_copy['jpg_url'].duplicated()]


# ### `name of all dogs predictions are variate with upper and lower case`

# ### Define

# Change all character in p1,p2, and p3 in lowercase

# ### Code

# In[119]:


img_pred_copy['p1'] = img_pred_copy['p1'].str.lower()
img_pred_copy['p2'] = img_pred_copy['p2'].str.lower()
img_pred_copy['p3'] = img_pred_copy['p3'].str.lower()


# ### Test

# In[120]:


img_pred_copy[['p1', 'p2', 'p3']]


# ### Test

# ### `retweeted tweets in json data`

# ### Define

# find retweeted tweets and delete it

# ### Code

# In[121]:


json_copy.head()


# In[122]:


#delete rows based on tweet id
for x in rt_tweet:
    rt_ind = json_copy[json_copy['tweet_id']== x].index
    json_copy.drop(rt_ind, inplace=True)


# ### Test

# In[123]:


# check if it still 'RT @' in full_text
cek_rt_cp = json_copy.full_text.str.extract(r'(RT @.*)')
cek_rt_cp[cek_rt_cp[0].notnull()]


# ## Cleaning data in Tidy

# ### `timestamp has date and time in one column`

# ### Define

# split timestamp into date and time 

# ### Code

# In[124]:


twit_arc_copy.timestamp


# In[125]:


# extract date and time from timestamp
twit_arc_copy['date'] = pd.DatetimeIndex(twit_arc_copy['timestamp']).date
twit_arc_copy['time'] = pd.DatetimeIndex(twit_arc_copy['timestamp']).time


# ### Test

# In[126]:


# check date and time validity
twit_arc_copy


# ### `text column contain text and url`

# ### Define

# extract text then store it to content column and extract url then store it to url column

# ### Code

# In[127]:


# get example of text in row 0
twit_arc_copy.text[30]


# In[128]:


###### extract content in each row
content_raw = twit_arc_copy.text.str.extract(r'(.*)(http|\n|$)')


# In[129]:


# droping https that is filtered by regex
content_raw[0][464]


# In[130]:


# extract url in content
content_url = twit_arc_copy.text.str.extract(r'(https://t.co/\w+)\ ?(https.*)?')


# In[131]:


# change column name
content_url.rename(columns={0:'url_content_1', 1:'url_content_2'}, inplace = True)


# In[132]:


# checkking are there any 2 urls in text column
twit_arc_copy.text[6]


# In[133]:


# check content_url
content_url


# In[134]:


# add parsed data into dataframe
twit_arc_copy['content'] = content_raw[0]
twit_arc_copy[['content_url_1', 'content_url_2']] = content_url


# In[135]:


twit_arc_copy


# ### Test

# In[136]:


# overview check parsed text
twit_arc_copy[['text', 'content', 'content_url_1', 'content_url_2']]


# In[137]:


# make sure there are no unparsed content
len(twit_arc_copy[twit_arc_copy.content.isnull()]['text'])


# ### Dropping unused data column

# ### Twitter archive

# In[138]:


# dropping timestamp and text
twit_arc_copy.drop(['timestamp', 'text'], axis = 1, inplace = True)


# timestamp and text are parsed, and the parsed data has been join with tweeter archive df. Dropping timestamp and text will reduce storage memory when data is stored.

# In[139]:


twit_arc_copy.drop(['retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp'], axis = 1, inplace = True)


# tweeter archive dataframe has been cleaned from retweeted tweet. In addition all column that have corelation with retweet only hold null value. retweeted_status_id, retweeted_status_user_id and retweeted_status_timestamp become unnecessary column based on previous reason.

# ### json data

# In[140]:


json_copy.head()


# In[141]:


json_copy.drop(['full_text', 'in_reply_to_status_id', 'in_reply_to_user_id'], axis = 1, inplace = True)


# full_text, in_reply_to_status_id and in_reply_to_user_id has stored in twitter archive data, storing it is unnecessary.

# ## Storing data

# In[142]:


# store data from df to csv and tsv
twit_arc_copy.to_csv('twitter_archive_master.csv', index = False)
img_pred_copy.to_csv('image_prediction_filtered.tsv', sep='\t', index = False)
json_copy.to_csv('favorite_retweet_filtered.csv', index = False)


# In[143]:


os.listdir()


# ## Recheck by opening data

# In[144]:


twit_ac = pd.read_csv('twitter_archive_master.csv')
image_filt =  pd.read_csv('image_prediction_filtered.tsv', sep = '\t')
rt_fav = pd.read_csv('favorite_retweet_filtered.csv')


# In[145]:


twit_ac.head(10)


# In[146]:


image_filt.head(50)


# In[147]:


rt_fav.head()


# ## data visualisation

# ### visualizing how often each predictor can predict image as a breed dog

# In[148]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[149]:


breed_predict = image_filt[['p1_dog', 'p2_dog', 'p3_dog']]


# In[150]:


breed_predict.sum().plot(kind='bar');
plt.title('How many times each predictor can predict image as dog breed')
plt.ylabel('frequency')


# The best three predictors can predict given image as dog breed with similar performance with small diferencies. 

# ### Visualizing mean true proporsion of dog breed prediction 

# In[151]:


#calculationg true and false prediction proportion
true_prop = breed_predict.sum().mean()/breed_predict.shape[0]
false_prop = 1 - true_prop


# In[152]:


# Pie chart of proporsion
labels = ['true', 'false']
sizes = [true_prop, false_prop]
colors = ['#90EE90','#FF0000']
explode = (0.05,0.05)
fig1, ax1 = plt.subplots()

plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', pctdistance=0.85, startangle=90, explode = explode)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax1.axis('equal')  
plt.tight_layout()
plt.title('true/false prediction of dog breed proporsion aggregation of all predictors')
plt.show()


# ### Normalize tweet rating 

# In[153]:


# ensure there are no 0 value in rating_denominator
twit_ac[twit_ac['rating_denominator']==0]


# Seeing that dog rating has different scales, normalization becomes very important to compare each dog rating. Then, normalization can be calculated by formula = rating_numerator / rating_denominator.
# 

# In[154]:


twit_ac['normalize_rating'].describe()


# In[155]:


twit_ac['normalize_rating'].hist(range=(0,2), bins=15);
plt.title('histogram of normalized dog rating')
plt.show()


# Normalized dog's rating data spreading is almost in normal distribution. There are some data outline that is proved by seeing describe data results. The max value of this parameter is 177.6 that is very far from mean data 1.22. Checking the data outline condition is needed to make data cleaner.

# In[156]:


twit_ac['normalize_rating'].sort_values(ascending=False)


# In[157]:


twit_ac.iloc[804,:]


# rating numerator is very big, let see the tweet content to prve this value.

# In[158]:


twit_ac.loc[804,'content']


# it is true that user has given that kind of score for the dog.Then to prove it in the unusual value, we need to print content on row 163, 162, 1895

# In[159]:


rows_sel = [162, 163, 1895]

for i in rows_sel:
    print(twit_ac.content[i])


# All data is valid by seeing content.

# ### RT and favorite count correlation

# In[160]:


rt_fav.head()


# In[161]:


plt.scatter(rt_fav['retweet_count'], rt_fav['favorite_count']);
plt.title('scatter plot retweet_count vs favorit_count')
plt.xlabel('retweet_count')
plt.ylabel('favorite_count')
plt.show()


# it seems that retweet_count and favorite_count has positive corellation. This assumption is based on scatter plot above. It can be seen that the more big retweet_count the more value favorite_count is. The correlation between two variables can be calculated in calculation method below.

# In[162]:


rt_fav[['retweet_count', 'favorite_count']].corr(method='pearson')


# Computed correlation between those variables is 0.927, which is proved that those variables has high enough positive correlation.

# In[ ]:




