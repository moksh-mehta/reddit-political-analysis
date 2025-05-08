import pandas as pd
import time
import json
import time
from datetime import datetime
from itertools import islice
import numpy as np
import os
import re
import networkx as nx
import ast
import community
import matplotlib.pyplot as plt
import sys
parent_dir = os.path.abspath('..')
data_utils_path = os.path.join(parent_dir, 'data')
sys.path.append(data_utils_path)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from data_utils import map_roots_to_subreddits
from itertools import groupby

#Check the political leanings of the textual data in the two political communities formed. 

#Mapping the textual data of all subreddits to their posts
def map_subreddits_to_posts(file_path="/Users/navyasahay/Desktop/Desktop - Navya's MacBook/Junior_year/Spring 2025/Data Science/final-projects-team-green/data/data/text_data.json"):
    '''
    Maps subreddits to all the texts in their posts
    '''
    # Get the data 
    with open(file_path, "r") as f:
        raw_data = json.load(f)


    subreddits_to_posts = {}
    for subreddit, posts in raw_data.items():
        texts = []
        for pair in posts: # Each pair is [title, body] of a post. Body can be Null
            clean_pair = []
            for part in pair:
                # Check for any null or null
                if isinstance(part, str):
                    if isinstance(part, str):
                        cleaned = re.sub(r'\\+|[\n\r\t]', ' ', part).strip()
                        if cleaned and cleaned.lower() != "null":
                            clean_pair.append(cleaned)
                if clean_pair:  # Only add if at least one part is valid
                    texts.append(" ".join(clean_pair)) 
        
        subreddits_to_posts[subreddit] = texts

    return subreddits_to_posts


seeds = ['conservative',
'politics',
'republican',
'liberal',
'democrats'
'progressive'
'joerogan'
'trump']

#Using this HuggingFace BERT-based model to get a political bias score:
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")

#Getting the textual data from the subreddits: 
subreddits_to_posts = map_subreddits_to_posts(file_path="/Users/navyasahay/Desktop/Desktop - Navya's MacBook/Junior_year/Spring 2025/Data Science/final-projects-team-green/data/data/text_data.json")

#Getting the distance of each subreddit node from the seed nodes 
roots_to_subreddits_distance = map_roots_to_subreddits()
communities_by_distance = {}

#Aggregrating the textual data for each community of subreddits based on distance
for k in roots_to_subreddits_distance.keys():
    communities_by_distance[k] =  list({key: list(group) for key, group in groupby(roots_to_subreddits_distance[k], key=lambda x: x[1])}.values())

centers_to_community_to_text = []

#For each community, taking the textual data and calculating the political bias -- ratio of left to right 
#based on the model output -- an array of size 3 where [0] = left percentage, [1] = percentage, [2] = right
#Note: the textual data of each community is broken to chunks due to the token limit of the model and then the 
# average of the logits of each chunk is taken 

for k in communities_by_distance.keys():
    center_text = subreddits_to_posts[k]
    community_texts = {}
    for c in communities_by_distance[k]:
        sub_text  = []
        for sub in c:
            chunk_logits = []
            dist = sub[1]
            if (sub[0] in subreddits_to_posts.keys()):
                add_text  = subreddits_to_posts[sub[0]]
            else:
                add_text = ""
            for i in range(0, len(add_text), 512):
                chunk = " ".join(add_text[i:i + 512])
                # Tokenizing just the chunk
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
                
                labels = torch.tensor([0])
                with torch.no_grad():
                    outputs = model(**inputs, labels=labels)
                
                chunk_logits.append(outputs.logits)
                # Averaging across all logits
                if chunk_logits:
                    mean_logits = torch.mean(torch.stack(chunk_logits), dim=0)
                    predictions = mean_logits.softmax(dim=-1)[0].tolist()
                    category = predictions[0] / predictions[2]
                                
                    sub_text.append(category)
        community_texts[dist] = sub_text
    centers_to_community_to_text.append([k, community_texts])


center_to_text_df = pd.DataFrame(centers_to_community_to_text)
center_to_text_df.to_csv("center_to_text.csv", index=False)