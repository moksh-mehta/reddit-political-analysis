TECH REPORT

Each individual value can be described as a subreddit1-subreddit2 relationship
that captures the overlap in users who posted from one to the other and
how much karma users of subreddit1 earned in subreddit2 to accurately weight
the relationship in analysis. While our original dictionary only has 524 values
as of writing, it captures 25,453 relationship datapoints that can be analyzed
to determine correlation. We believe that 25,453 total datapoints, with up to 50
relationships per row, is more than enough data to determine subreddit 
correlations.

The identifying attributes of our data are outlined in the main README, with
each datapoint between two subreddits being accessible by querying one of the
"main" subreddits in the first dictionary and then querying the adjacency
dictionary for that subreddit. It isn't possible to query individual pairings 
directly, though this is intentional as we plan to only do clustering on the
~500 subreddits we explicitly queried using the 25,000 relationship datapoints.

Our data is all derived from Reddit, and was collected with the Python Reddit
API Wrapper, or PRAW. We registered our script with Reddit to get an API key
and followed all their policies on API calls, implementing checks to make sure
we did not go over the call limit. Given we are scraping directly from Reddit
this is the most reputable source possible.

To collect our data we first defined a handful of "seed" subreddits as listed
in seeds.txt and used a breadth-first search to add new subreddits to a queue
that the script worked through gradually. A subreddit was added to the queue
if the one the script was currently on was found to have at least 5 users
who all participated in that community as well, with a limit of three additions
per subreddit to prevent excessive queues. Subreddits also had to have at
least 50,000 members to merit a search.

It is worth noting that these "users" for each subreddit were not selected out
of the whole subreddit as community membership on Reddit is private. Instead,
users were taken from the top 10 posts over the past year, month, week, and day
at the time of collection to give a representative sample of the users in the
community. This obviously introduces bias as it only selects the top users in
a community, but we felt this was actually a good thing as those top users would
be the best representation of the subreddit.

Since Reddit also makes private what posts and comments users have upvoted,
we instead looked at the top and newest 50 posts and comments for each user,
so 200 pieces of content total. While there is some potential for overlap here,
for example if the users most recent post was also their top one, we determined
this wouldn't run counter to our goal of finding the other communities a 
subreddit's top users were active on. 

Using these 200 pieces of content the total karma score and community name were
added to a subdictionary for the overarching community being analyzed in that
loop iteration, and counters were used to keep track of how much overlap
there was with each community.

All the filtering parameters described above are modifiable and explained in 
the main script. The values we used to get our sample are as follows:
per_sub = 10
per_user = 50
threshold = 5
max_subs = 1000
min_members = 50000
max_members = 100000000
max_sub_communities = 50
branch_limit = 3

These values were all used as they achieved a balance of our limited resoureces,
which were mostly time and API calls. While we could have gotten every piece of
content for every user in the past year that would have taken forever and would
by no means be feasible, so we used a small convenience sample for each
community.

Our sample of ~500 subreddits is only a fraction of all the communities on
Reddit, though due to how our selection process works it covers many of the
largest communities on the platform. The top communities would have the most
overlap amongst users in the seed communities, and they were quickly added
to the queue as a result.

All of our data was cleaned as we were collecting it, meaning no cleaning was 
done after the JSON file was updated. We checked that users were not suspended
as that would lead to issues fetching their content, and included many checks
on duplication for both the queue and the authors we looked at. Due to all this
live checking and parsing there were no real issues with data cleanliness after
the script was finished running, though we did have to manually remove some 
communities that were NSFW and unrelated to our project goals.

The main challenge throughout the data collection process was ensuring we had
an accurate sample for each subreddit while balancing system resources and API
calls. We used the methods above to filter which authors we would delve into
for each community to save API calls as we only got 1000 every 5 minutes or so,
and each author would take 4 api calls each. We also limited the content we
looked at for each author as some users had thousands of posts and comments
that slowed the entire program down signficiantly and didn't add much to our
sample. Manual checking proved key to solving the API issues, and a formula
was devised to wait for more API calls if the next series of requests would
go over the limit.

Because the data we collected focuses on subreddit correlations we will need
to use some sort of clustering or covariance related machine learning model to
analyze it. We haven't settled on what this will be yet, but we are confident
that we have enough data to put together a solid analysis of some of the top
communities on Reddit, both political and not.