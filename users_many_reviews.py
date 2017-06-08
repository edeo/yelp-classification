import json

with open('user_review_dictionary.json') as json_data:
    d = json.load(json_data)

mongo_biguser =[]

for user in d.keys():
      if len(d[user]) > 100:
        mongo_biguser.append(user)

users= {"user_ids": mongo_biguser }

with open('users_many_reviews_dictionary.json', 'w') as outfile:
    json.dump(users, outfile)
