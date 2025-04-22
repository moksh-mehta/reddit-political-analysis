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
print(communities)


graph_layout = nx.spring_layout(graph)

colors = plt.cm.get_cmap('tab20',len(communities))

for i,com in enumerate(communities):
    nx.draw_networkx_nodes(graph, graph_layout, nodelist=list(com),node_color=[colors(i)],label=f"Community{i}", node_size=50)

nx.draw_networkx_edges(graph, graph_layout, alpha=0.2)

plt.title("Community clusters")
plt.axis("off")
plt.legend(markerscale=3)
plt.show()
