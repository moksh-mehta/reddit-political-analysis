from data import data_utils as utils


# Get root subreddits to all subreddits
file_path ="relations.json" 
roots = {"conservative","politics","republican","liberal","democrats","progressive","joerogan","trump"}
roots_to_subreddits = utils.map_roots_to_subreddits(file_path=file_path, roots=roots)

# Get the sentiment of every subreddit