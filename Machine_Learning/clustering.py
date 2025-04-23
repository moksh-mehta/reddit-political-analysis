import pandas as pd
import time
import json
import time
from datetime import datetime
from itertools import islice
import os
import networkx as nx
import ast
import community
import matplotlib.pyplot as plt
import sys
parent_dir = os.path.abspath('..')
data_utils_path = os.path.join(parent_dir, 'data')
sys.path.append(data_utils_path)
from data_utils import map_subreddits_to_posts
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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
    text = ''
    for item in com:
        if (item not in seeds) and (item in subreddits_to_posts):
            sub_posts = subreddits_to_posts[item]
            for s in sub_posts:
                text.join(s)

    text_by_community.append(text)


 

for t in text_by_community:
    inputs = tokenizer(t, return_tensors="pt")
    labels = torch.tensor([0])
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]
    print(logits.softmax(dim=-1)[0].tolist()) 








