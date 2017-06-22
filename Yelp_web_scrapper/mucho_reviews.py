
# coding: utf-8

# # Yelp Web Scrapper for restaurants in Austin, TX

# In[2]:

from bs4 import BeautifulSoup
import requests
import re
import json
import scrapping_functions as sf
# may need pip install urllib3
import urllib3

urllib3.disable_warnings()
target_url = 'https://www.yelp.com/search?find_desc=Restaurants&find_loc=Austin+TX&start='
base = 'http://www.yelp.com'


# In[3]:

#Pull in a list of links from the target url
link_dict = {}
for x in range(10, 990, 10):
    target = target_url + str(x)
    raw_html = requests.get(target, verify=False)
    soup = BeautifulSoup(raw_html.text, 'html.parser')
    link_dict = sf.biz_links(soup, link_dict)

print("\nFinished!")


# In[4]:

#Write all links to a text file
biz_links = open('cleanbiz_austin_links.txt', 'w')
for item in link_dict.keys():
    if "adredir" in item: 
        continue
    print("key is: " + item)
    biz_links.write("%s\n" % item)
biz_links.close()
print("\nFinished writing to file:  cleanbiz_austin_links.txt")


# We would like to construct the following dictionary:
# 
# biz_dict = {biz_name: {"city": "Washington", "state": "DC", "category_aliases": "newamerican,breakfast_brunch", "biz_id": "wO-7cBBOYUdiLflpuRsu9A", "latitude": 38.90842, "biz_name": "The Bird", "city_state": "Washington, DC", "longitude": -77.026685, "geoquad": 12845454}}
# 
# unique_id = 5D32F13B349CE2AD
# 
# #### Algorithm design:
# 
# For each link in biz_links:
#     > set biz_name = replace("_", '-' in link)
#     > find the unique_id
#     > assign biz_dict[biz_name] = dictionary(biz_name)

# In[5]:

link_file = open("cleanbiz_austin_links.txt", "r")
link_list = link_file.read().split('\n')
link_list = list(set(link_list))
for link in link_list:
    if link == '':
        link_list.pop(link_list.index(link))


# In[ ]:




# In[6]:

biz_dict = {}

for biz_name in link_list:
    biz_dict[biz_name] = {}
    raw_html = requests.get(base + "/biz/the-lucky-belly-austin", verify=False)
    print(base + biz_name)
    soup = BeautifulSoup(raw_html.text, 'html.parser')
    biz_dict[biz_name] = json.loads(soup.find('script', type='application/ld+json').text)
    
print("\nFinished")


# In[7]:

#Output JSON file of all the review details
with open('austin_reviews.json', 'w') as outfile:
    json.dump(biz_dict, outfile)
print("Finished")


# In[23]:

raw_html = requests.get("https://www.yelp.com/biz/franklin-barbecue-austin?start=160", verify=False)


# In[16]:

raw_html


# In[24]:

soup = BeautifulSoup(raw_html.text, 'html.parser')


# In[18]:

soup


# In[25]:

soup.find('script', type='application/ld+json').text


# In[ ]:



