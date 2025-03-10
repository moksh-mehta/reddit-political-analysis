# Use PRAW to access Reddit info
# See docs for information: https://praw.readthedocs.io/en/latest/code_overview/reddit_instance.html
# https://praw.readthedocs.io/en/latest/code_overview/models/redditor.html

import praw
import pandas as pd
import requests
import time
import json
import time
from datetime import datetime

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
# Number of posts to scan per subreddit
count = 10

# Subreddit info will be stored as a dataframe with:
# top_subs, a dict with ["subreddit": karma]
# size
cols = ["sub_name", "top_subs", "size"]
subs = pd.DataFrame(columns=cols)

# Iterate through subreddits
# Each sub uses 1 + (3 * count) PRAW calls
for sub in subreddits:
    # These are at top to trigger PRAW api calls info
    subreddit = reddit.subreddit(sub)
    subscribers = reddit.subreddit(sub).subscribers
    # Make sure we have enough calls, otherwise wait
    auth_limits = reddit.auth.limits
    print(auth_limits)
    # Keep track of how many calls each thing takes
    if auth_limits and auth_limits["remaining"]:
        left = auth_limits["remaining"]
        # If we would run out of calls wait until reset
        # This will need to be modified, currently 2 calls per author and 2
        # per subreddit
        if left - 2 - (2 * count) < 0:
            timestamp = time.time()
            reset_time = auth_limits["reset_timestamp"]
            difference = reset_time - timestamp
            # Sleep until we would reset
            print(f"Waiting {int(difference)} seconds for more API calls!")
            # Sleep if there is a difference
            if (reset_time - timestamp > 0):
                time.sleep(reset_time - timestamp)
    elif count > 999:
        print("Post limit is 999 per subreddit due ot API limits!")
    # Grab all authors
    authors = []
    for submission in subreddit.top(time_filter = time_limit, limit=count):
        authors.append(submission.author)
    # Iterate through authors, getting their top subreddits and their karma
    top_subs = {}
    for author in authors:
        # Feedback mechanism for large requests
        print(author)
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
    new_row = (sub, sortable, subscribers)
    subs.loc[len(subs)] = new_row
    print(str(reddit.auth.limits["remaining"]) + " calls left!")
print(subs)
# Save
subs.to_csv('query.csv', index=False)