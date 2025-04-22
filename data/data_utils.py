import json
from collections import deque, defaultdict

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def map_roots_to_subreddits(file_path="./data/relations.json" , roots={"conservative","politics","republican","liberal","democrats","progressive","joerogan","trump"}):
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

    for root in roots:
        subreddit_to_origin[root] = root
        queue.append(root)

    while queue:
        curr_subreddit = queue.popleft()
        neighbours = raw_data.get(curr_subreddit, [{}, 0])[0].keys()

        for neighbour in neighbours:
            if neighbour not in subreddit_to_origin:
                subreddit_to_origin[neighbour] = subreddit_to_origin[curr_subreddit]
                queue.append(neighbour)

    
    # Convert subreddit -> origin to origin -> subreddit
    root_to_subreddits = defaultdict(set)
    for subreddit, root in subreddit_to_origin.items():
        root_to_subreddits[root].add(subreddit)

    return root_to_subreddits

def map_subreddits_to_posts(file_path="./data/text_data.json"):
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


def get_indidual_sentiment_score(text: str) -> float:
    '''
    Returns the sentiment score of a single string. Sentiment is waited
    from -1 to 1, where -1 is very negative and 1 is very positive.
    '''
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()

    # Model outputs 3 probs: [neg, neutral, pos]. Score is expected value of all 3
    return -1 * probs[0].item() + 0 * probs[1].item() + 1 * probs[2].item()



def get_sentiment_scores(file_path="./data/text_data.json"):
    '''
    Returns a map from every subreddit to their average sentiment. Sentiment is waited
    from -1 to 1, where -1 is very negative and 1 is very positive.

    I am allowing a 
    '''

    subreddits_to_posts = map_subreddits_to_posts(file_path)

    