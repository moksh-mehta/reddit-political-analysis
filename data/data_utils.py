import json
from collections import deque, defaultdict

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def map_roots_to_subreddits(file_path="/Users/navyasahay/Desktop/Desktop - Navya's MacBook/Junior_year/Spring 2025/Data Science/final-projects-team-green/data/data/relations.json" , roots={"conservative","politics","republican","liberal","democrats","progressive","joerogan","trump"}):
    """
    Converts the reddit overlap graph into a dictionary from the original subreddit roots
    to all the subreddits they connect to.

    Example output:
    {joe_rogan: {whitePeopleTwitter, dogLovers, hello},
     trump: {12344, example_example},
     ...
    }
    """
    # Load Data
    with open(file_path, "r") as f:
        raw_data = json.load(f)

    # BFS On the Data JSON
    subreddit_to_origin = {}
    queue = deque()

    seeds_distances = {}

    for root in roots:
        subreddit_to_origin[root] = root
        seeds_distances[root] = 0
        queue.append(root)

    while queue:
        curr_subreddit = queue.popleft()
        neighbours = raw_data.get(curr_subreddit, [{}, 0])[0].keys()
        current_distance = seeds_distances[curr_subreddit]

        for neighbour in neighbours:
            if neighbour not in subreddit_to_origin:
                subreddit_to_origin[neighbour] = subreddit_to_origin[curr_subreddit]
                seeds_distances[neighbour] = current_distance + 1
                queue.append(neighbour)

    
    # Convert subreddit -> origin to origin -> subreddit
    root_to_subreddits = defaultdict(set)
    for subreddit, root in subreddit_to_origin.items():
        root_to_subreddits[root].add((subreddit, seeds_distances[subreddit]))

    return root_to_subreddits

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
            for part in pair:
                # Check for any null or null
                if isinstance(part, str) and part.strip():
                    texts.append(part)
        
        subreddits_to_posts[subreddit] = texts

    return subreddits_to_posts

MODEL_NAME = "Muddassar/longformer-base-sentiment-5-classes"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def get_indidual_sentiment_score(text: str) -> float:
    if not text.strip():
        return 0.0
    
    '''
    Returns the sentiment score of a single string. Sentiment is waited
    from -1 to 1, where -1 is very negative and 1 is very positive.
    '''
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()

    # Map 5 class sentiment onto [-1,1]
    weights = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=probs.device)
    score = (probs * weights).sum().item()
    return round(score, 4) 



def get_sentiment_scores(file_path="/Users/navyasahay/Desktop/Desktop - Navya's MacBook/Junior_year/Spring 2025/Data Science/final-projects-team-green/data/data/text_data.json"):
    '''
    Returns a map from every subreddit to their average sentiment. Sentiment is waited
    from -1 to 1, where -1 is very negative and 1 is very positive.

    I am allowing a 
    '''
    subreddits_to_average_sentiment = {}

    subreddits_to_posts = map_subreddits_to_posts(file_path)
    print("Finished mapping subreddits to posts")

    for subreddit, texts in subreddits_to_posts.items():
        print(f"Subreddit: {subreddit}")
        total_sentiment = 0
        for text in texts:
            total_sentiment += get_indidual_sentiment_score(text)

        subreddits_to_average_sentiment[subreddit] = total_sentiment / len(texts)

    return subreddits_to_average_sentiment



        
    