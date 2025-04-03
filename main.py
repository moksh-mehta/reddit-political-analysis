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
from itertools import islice

reddit = praw.Reddit(
    client_id="q33sziEQUMFF51P9IrtRHw",
    client_secret="xxNns4HmThugUzy6-o8LqsF8IY9W9A",
    user_agent="MacOS:q33sziEQUMFF51P9IrtRHw:1.0.3",
)

# Get target subreddits
inputs = open("subs.txt", "r")
data = inputs.read()
subreddits = data.split("\n")
subreddits = [item.lower() for item in subreddits]

# PARAMETERS
# Which times to look for top posts
times = ["year", "month", "week", "day"]
# Number of posts to scan per subreddit
# Impacts API call amount
# Categories are top for year, month, week, day
# THIS IS x4 as a result
per_sub = 10
# Number of items per category to get for each user
# Does not impact API call amount but is multiplied by 4
per_user = 50
# Threshold for minimum member crossover to merit a deeper search
threshold = 5
# Max subreddit to search, limit time
max_subs = 1000
# Minimum subscribers for a subreddit to be searched
min_members = 50000
# Max subscribers for a subreddit to be searched, limit irrelevant ones
max_members = 100000000
# Define a limit for how many sub-communities there should be
max_sub_communities = 50
# Define a limit for branching, i.e. only top 10
branch_limit = 3

# Subreddit info will be stored as a dataframe with:
# top_subs, a dict with ["subreddit": karma]
# size
cols = ["sub_name", "top_subs", "size"]
subs = pd.DataFrame(columns=cols)

# Output dict for json
output = {}

# Collect post text for sentiment analysis
text_data = {}

# Iterate through subreddits
# Each sub uses times + (times * per_sub * 4) calls
# So year, month, week, day with 10 authors per sub is 164 calls per sub
for sub in subreddits:
    # Store post body text data in a list
    raw_text = []
    # These are at top to trigger PRAW api calls info
    subreddit = reddit.subreddit(sub)
    subscribers = reddit.subreddit(sub).subscribers
    # Skip if not enough subs
    if subscribers < min_members or subscribers > max_members:
        continue
    # Make sure we have enough calls, otherwise wait
    auth_limits = reddit.auth.limits
    print(auth_limits)
    # Keep track of how many calls each thing takes
    if auth_limits and auth_limits["remaining"]:
        left = auth_limits["remaining"]
        # If we would run out of calls wait until reset
        # This will need to be modified, currently 4 calls per author and 2
        # per subreddit
        if left - len(times) - (len(times) * 4 * per_sub) - 2 < 0:
            timestamp = time.time()
            reset_time = auth_limits["reset_timestamp"]
            difference = reset_time - timestamp
            # Sleep until we would reset
            print(f"Waiting {int(difference)} seconds for more API calls!")
            # Sleep if there is a difference
            if (reset_time - timestamp > 0):
                time.sleep(reset_time - timestamp)
    elif per_sub > 999:
        print("Post limit is 999 per subreddit due ot API limits!")
    # Grab all authors
    authors = []
    # Do year, month, week, day
    for item in times:
        for submission in subreddit.top(time_filter = item, limit=per_sub):
            title = submission.title
            text = submission.selftext
            if len(text) > 0:
                raw_text.append((title, text))
            else:
                raw_text.append((title, None))
            authors.append(submission.author)
    # Iterate through authors, getting their top subreddits and their karma
    top_subs = {}
    for author in set(authors):
        # Keep track of their subreddits
        my_subs = []
        # Feedback mechanism for large requests
        if author is not None:
            # Use praw to get the content of this user - avoids requests issue
            content = []
            # Add comments and posts from both top and new
            # Use try/except in case of a banned/private account
            try:
                content.extend(reddit.redditor(author.name).comments.top(limit = per_user))
                content.extend(reddit.redditor(author.name).submissions.top(limit = per_user))
                content.extend(reddit.redditor(author.name).comments.new(limit = per_user))
                content.extend(reddit.redditor(author.name).submissions.new(limit = per_user))
            except:
                pass
            # Get top/newwest comments
            for item in set(content):
                # Get name of subreddit
                display = item.subreddit.display_name
                if display.lower().rstrip() != subreddit.display_name.lower().rstrip():
                    # Add score
                    if display in top_subs:
                        top_subs[display][0] += item.score
                        if display not in my_subs:
                            # Add to member counter
                            top_subs[display][1] += 1
                            my_subs.append(display)
                    else:
                        top_subs[display] = [item.score, 1]
                        my_subs.append(display)
        # Add raw text to dictionary
    text_data[sub] = raw_text
    print(raw_text)
    time.sleep(3)
    # Fitler by shred subreddits
    new_top = {k: v for k, v in top_subs.items() if v[1] >= 2}
    trimmed_top = dict(islice(new_top.items(), max_sub_communities))
    sortable = dict(sorted(trimmed_top.items(), key=lambda item: item[1][0], reverse=True))
    # Add crawling too, this uses BFS to go down layers
    # Only get first branch_limit ones
    count = 0
    for k, v in sortable.items():
        if count < branch_limit:
            if v[1] >= threshold and k not in subreddits and max_subs > len(subreddits):
                subreddits.append(k)
                count += 1
    print(subreddits)
    output[sub] = (sortable, subscribers)
    # Dump as json file
    with open("relations.json", 'w') as file:
        json.dump(output, file, indent = 4)
    # Dump raw text as well
    with open("text_data.json", 'w') as file:
        json.dump(text_data, file, indent = 4)
    new_row = (sub, sortable, subscribers)
    subs.loc[len(subs)] = new_row
    # Also save csv
    subs.to_csv('relations.csv', index=False)
    print(str(reddit.auth.limits["remaining"]) + " calls left!")