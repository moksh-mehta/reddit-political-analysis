# final-projects-team-green

DATA SPEC:
Our data is all scraped from Reddit using the Python Reddit API Wrapper, or
PRAW. We followed all registration protocols and used a valid API key along
with informative headers as per Reddit policy.

Our data is formatted in both CSV and JSON format, and takes the format of
a dictionary. The main dictionary links subreddits, by name, to a list that has
a dictionary and an integer. The integer is the subreddit's size, and the other
dictionary has adjacent subreddits as the key and a list of adjacency info
as a value. This list has only two elements, the total karma users in the
original subreddit earned posting/commenting on that one, and the total overlap
in terms of users who posted/commented in both communities.

This is the general structure of the data:
{"Original-sub": [{"sub1": [sub1-karma, user-overlap1], 
                  "sub2": [sub2-karma, user-overlap2]},
                  original-size]}

This structure was used as it is the most compact way to store the necessary
adjacency information, and dictionaries are easily representable in JSON format.

Because this data was scraped off Reddit and error handling was done during
the scraping process there are no blank values, and all the filtering was
done beforehand to save API calls down the line. There are no default values
as the dictionary was constructed only with the valid data that was returned
by the API, and checks were done to catch any duplicates before they were added.

Using the example strings in the above JSON example as a reference:

original-sub - a string representing the subreddit from which authors were 
               collected from. For each author their top and newest posts and
               comments were scraped to determine what communities they
               interacted with.

sub1, sub2 - these two placeholder strings represent subreddits that authors
             who posted in original-sub also posted/commented in.

sub1-karma - this integer represents the total karma authors who posted in 
             original-sub got from posts and/or comments in sub1. This is taken
             from the newest and top n comments and posts, where n is a
             user-defined limit to standardize contributions across users.

user-overlap1 - this integer represents how many users from the original
                subreddit also posted/commented in sub1. It is used to determine
                how strong the relationship is, along with the above karma value

original-size - this integer is just the size of the original subreddit, for
                weighting purposes later on.

DATA LINK/SAMPLE:
https://raw.githubusercontent.com/csci1951a-spring-2025/final-projects-team-green/refs/heads/main/huge_output.json?token=GHSAT0AAAAAAC5X6QY6ZIY4DFVRT2RZUYZWZ63GIIA

Given this is a plaintext JSON file there should be no issues downloading this 
or viewing individual datapoints, so a smaller sample is not necessary.