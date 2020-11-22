import csv
import numpy as np

# The domesticviolence and survivorsofabuse subreddits will be class 0, critical; these are personal stories, calls for help, requests for advice
# The abuseInterrupted subreddit will be class 1, noncritical; it mostly contains empty text, links to articles, general statements about abuse, etc.
# Everything else will be class 2, general/unrelated
CLASSES = {
    'relationship_advice': 2, # 5874 samples
    'relationships': 2, # 8201 samples
    'casualconversation': 2, # 7286 samples
    'advice': 2, # 5913 samples
    'anxiety': 2, # 4183 samples
    'anger': 2, # 837 samples
    'abuseinterrupted': 1, # 1653 samples
    'domesticviolence': 0, # 749 samples
    'survivorsofabuse': 0 # 512 samples
}

def load_data(file_path):
    """
    Loads text and labels from dataset stored in file_path, a CSV.
    """
    posts = []
    labels = []

    with open(file_path) as f:
        reader = csv.reader(f)
        # Skip the first row that just has column names
        rows = list(reader)[1:]
        for row in rows:
            # print("\n" + str(row))
            subreddit_name = row[0]
            label = CLASSES[subreddit_name]
            post_title = row[2]
            post_text = row[3]
            post_title_and_text = post_title + " " + post_text
            labels.append(label)
            posts.append(post_title_and_text)

    return (np.array(posts), np.array(labels))