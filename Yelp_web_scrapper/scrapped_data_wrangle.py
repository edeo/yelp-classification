reviews_json = json.load(open('newyork_reviews.json'))

#creates a dictionary where the key is the restaurant name and the value is a list of all the reviews for that restaurant

biz_w_list_of_reviews = {}

for review in reviews_json.keys():
    biz_w_list_of_reviews[review]=[]
    for text in reviews_json[review]['review']:
      biz_w_list_of_reviews[review].append(text)
      
      
 
