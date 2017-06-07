import json

with open('user_review.json') as json_data:
    d = json.load(json_data)

mongo_biguser =[]

for user in d.keys():
      if len(d[user]) > 100:
        mongo_biguser.append(user)

users= {"user_ids": mongo_biguser }
