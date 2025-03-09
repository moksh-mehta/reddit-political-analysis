# Use PRAW to access Reddit info
# See docs for information: https://praw.readthedocs.io/en/latest/code_overview/reddit_instance.html
# https://praw.readthedocs.io/en/latest/code_overview/models/redditor.html

import praw
import pandas as pd
import requests
import time
import json

reddit = praw.Reddit(
    client_id="q33sziEQUMFF51P9IrtRHw",
    client_secret="xxNns4HmThugUzy6-o8LqsF8IY9W9A",
    user_agent="userinfo",
)

# List target subreddits
subreddits = ["politics", "news", "socialism"]
subreddits = [item.lower() for item in subreddits]

# Set some parameters
time_limit = "month"
count = 30

# Subreddit info will be stored as a dataframe with:
# top_subs, a dict with ["subreddit": karma]
# size
cols = ["sub_name", "top_subs", "size"]
subs = pd.DataFrame(columns=cols)


# Iterate through subreddits
for sub in subreddits:
    # Grab all authors

    subreddit = reddit.subreddit(sub)
    authors = []
    for submission in subreddit.top(time_filter = time_limit, limit=count):
        authors.append(submission.author)
    # Iterate through authors, getting their top subreddits and their karma
    top_subs = {}
    for author in authors:
        if author != None:
            # Use praw to get the content of this user - avoids requests issue
            content = []
            # DESIGN QUESTION: Do we want top or most recent posts/comments
            # Add comments and posts
            content.extend(reddit.redditor(author.name).comments.top(limit = 10))
            content.extend(reddit.redditor(author.name).submissions.top(limit = 10))
            # Get top/newwest comments
            for item in content:
                # Add score
                if item.subreddit in top_subs:
                    top_subs[item.subreddit] += item.score
                else:
                    top_subs[item.subreddit] = item.score
    print(top_subs)
    sortable = dict(sorted(top_subs.items(), key=lambda item: item[1], reverse=True))
    new_row = (sub, sortable, subreddit.subscribers)
    subs.loc[len(subs)] = new_row
print(subs)
# Save
subs.to_csv('query.csv', index=False)