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

data_path = "/Users/navyasahay/Desktop/Desktop - Navya's MacBook/Junior_year/Spring 2025/Data Science/final-projects-team-green/data/data/relations.csv"
subs = pd.read_csv(data_path)

graph = nx.Graph()

for index, row in subs.iterrows():
    subreddit = row['sub_name']
    sub_dictionary = ast.literal_eval(row['top_subs'])
    graph.add_node(subreddit)

    for sub in sub_dictionary.keys():
        karma_count, user_count  = sub_dictionary[sub]
        edge_weight = karma_count / user_count
        if (user_count > 5):
            graph.add_node(sub)
            graph.add_edge(subreddit, sub, weight=edge_weight)


communities = nx.community.louvain_communities(graph)
#print(communities)
political_communities = []
for com in communities:
    if ('conservative' in com) or ('liberal' in com) or ('politics' in com) or ('republican' in com) or ('democrats' in com) or ('trump' in com) or ('worldnews' in com) or ('progressive' in com):
        political_communities.append(com)
        

graph_layout = nx.spring_layout(graph)

colors = plt.cm.get_cmap('tab20',len(political_communities))

for i,com in enumerate(political_communities):
    nx.draw_networkx_nodes(graph, graph_layout, nodelist=list(com),node_color=[colors(i)],label=f"Community{i}", node_size=50)

nx.draw_networkx_edges(graph, graph_layout, alpha=0.2)

plt.title("Community clusters")
plt.axis("off")
plt.legend(markerscale=3)
plt.show()


#Check the political leanings of the textual data in the two political communities formed. 

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

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")

subreddits_to_posts = map_subreddits_to_posts(file_path="/Users/navyasahay/Desktop/Desktop - Navya's MacBook/Junior_year/Spring 2025/Data Science/final-projects-team-green/data/data/text_data.json")
text_by_community = []

for com in political_communities:
    text = " "
    for item in com:
        if (item not in seeds) and (item in subreddits_to_posts):
            sub_posts = subreddits_to_posts[item]
            text += " ".join(sub_posts) 

    text_by_community.append(text)

 
'''
for t in text_by_community:
    inputs = tokenizer(t, return_tensors="pt")

    chunk_logits = []
    for i in range(0, inputs['input_ids'].shape[1], 512):
        labels = torch.tensor([0])
        # Slice the input to create a chunk of size max_length
        chunk_input_ids = inputs['input_ids'][0][i:i + 512].unsqueeze(0)
        chunk_attention_mask = inputs['attention_mask'][0][i:i + 512].unsqueeze(0) if 'attention_mask' in inputs else None

        # Prepare chunk inputs
        chunk_inputs = {
            'input_ids': chunk_input_ids,
            'attention_mask': chunk_attention_mask
        }
        
        # Pass the chunk through the model
        with torch.no_grad():
            chunk_outputs = model(**chunk_inputs,labels=labels)
        
        # Collect the logits for each chunk
        chunk_logits.append(chunk_outputs.logits)

    # Stack the logits and compute the mean across all chunks
    mean_logits = torch.mean(torch.stack(chunk_logits), dim=0)

    
  
    print(mean_logits.softmax(dim=-1)[0].tolist()) 

'''

roots_to_subreddits_distance = map_roots_to_subreddits()
communities_by_distance = {}


for k in roots_to_subreddits_distance.keys():
    communities_by_distance[k] =  list({key: list(group) for key, group in groupby(roots_to_subreddits_distance[k], key=lambda x: x[1])}.values())

centers_to_community_to_text = []

for k in communities_by_distance.keys():
    center_text = subreddits_to_posts[k]
    community_texts = {}
    for c in communities_by_distance[k]:
        text = " "
        chunk_logits = []
        for sub in c:
            dist = sub[1]
            if (sub[0] in subreddits_to_posts.keys()):
                add_text  = subreddits_to_posts[sub[0]]
                text += " ".join(add_text)
            else:
                add_text = ""
                text += ""
        
        inputs = tokenizer(text, return_tensors="pt")
        for i in range(0, inputs['input_ids'].shape[1], 512):
            labels = torch.tensor([0])
            # Slice the input to create a chunk of size max_length
            chunk_input_ids = inputs['input_ids'][0][i:i + 512].unsqueeze(0)
            chunk_attention_mask = inputs['attention_mask'][0][i:i + 512].unsqueeze(0) if 'attention_mask' in inputs else None

            # Prepare chunk inputs
            chunk_inputs = {
                'input_ids': chunk_input_ids,
                'attention_mask': chunk_attention_mask
            }
            
            # Pass the chunk through the model
            with torch.no_grad():
                chunk_outputs = model(**chunk_inputs,labels=labels)
            
            # Collect the logits for each chunk
            chunk_logits.append(chunk_outputs.logits)

    # Stack the logits and compute the mean across all chunks
        mean_logits = torch.mean(torch.stack(chunk_logits), dim=0)
        mean_logits = torch.mean(torch.stack(chunk_logits), dim=0)
        predictions = mean_logits.softmax(dim=-1)[0].tolist()
        category = torch.argmax(torch.tensor(predictions))
        community_texts[dist] = category
    centers_to_community_to_text.append([k, community_texts])


center_to_text_df = pd.DataFrame(centers_to_community_to_text)
center_to_text_df.to_csv("center_to_text.csv", index=False)








