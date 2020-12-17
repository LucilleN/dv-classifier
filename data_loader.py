"""
This file is used for two purposes: to (1) load the data from existing CSV's in order for our 
models to train/test on, and (2) when run independently, to generate new augmented data.

The `load_data` file is imported into the model files and run within their scripts.

To generate augmented data, run this file with the --generate_augmented_data flag and optionally
specify the following additional flags:
    --clear_old_augmented_data: If set, will overwrite the old augmented data rather than appending to it
    --num_new_class_0: The number of samples of class 0 to generate
    --num_new_class_1: The number of samples of class 1 to generate
    --write_to_path: The file path to write or append the new data to

Example Usage:
    python3 data_loader.py --generate_augmented_data --num_new_class_0 3 --num_new_class_1 3 --write_to_path data/temp.csv
"""

import csv
import numpy as np
import random

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import argparse

import nltk


# - The domesticviolence and survivorsofabuse subreddits will be class 0, critical; these are personal stories,
# calls for help, requests for advice.
# - The abuseInterrupted subreddit will be class 1, noncritical; it mostly contains empty text, links to
# articles, general statements about abuse, etc.
# - Everything else will be class 2, general/unrelated
CLASSES = {
    'relationship_advice': 2,  # 5874 samples
    'relationships': 2,  # 8201 samples
    'casualconversation': 2,  # 7286 samples
    'advice': 2,  # 5913 samples
    'anxiety': 2,  # 4183 samples
    'anger': 2,  # 837 samples
    'abuseinterrupted': 1,  # 1653 samples
    'domesticviolence': 0,  # 749 samples
    'survivorsofabuse': 0  # 512 samples
}

CLASS_COUNTS = {
    2: 32294,
    1: 1653,
    0: 1261
}

LABEL_TO_IX = {
    'critical': 0,
    'noncritical': 1,
    'general': 2
}

IX_TO_LABEL = {
    0: 'critical',
    1: 'noncritical',
    2: 'general'
}


def load_data(
        og_file_path,
        aug_file_path=None,
        include_og=True,
        include_aug=False,
        fraction_class_2_to_load=1.0,
        combine_classes_01=False):
    """
    Loads text and labels from dataset stored in the specified file_path(s), which must be CSV's,
    and returns a tuple of two parallel numpy arrays, one that contains the raw post strings,
    and one that contains the labels for each post.
    Takes in:
    - og_file_path: The file path where the original data is stored
    - aug_file_path: The file path where the augmented data is stored
    - include_og: Boolean representing whether or not to include the original data 
    - fraction_class_2_to_load: A float representing the fraction of class 2 to include 
    - combine_classes_01: Boolean representing whether to combine the critical and noncritical classes 
    into one general class for anything DV-related
    """
    posts = []
    labels = []

    sources = []
    if include_og:
        sources.append(og_file_path)
    if include_aug:
        sources.append(aug_file_path)

    class_2_max = int(fraction_class_2_to_load * CLASS_COUNTS[2])

    for file_path in sources:

        with open(file_path) as f:
            reader = csv.reader(f)

            # Skip the first row that just has column names
            rows = list(reader)[1:]
            for row in rows:
                subreddit_name = row[0]
                label = CLASSES[subreddit_name]
                if combine_classes_01:
                    # If we're combining the critical and noncritical classes, than labels 0 or 1 will be
                    # collapsed into label 0, and label 2 will become label 1
                    label = 0 if label < 2 else 1
                post_title = row[2]
                post_text = row[3]
                post_title_and_text = post_title + " " + post_text
                labels.append(label)
                posts.append(post_title_and_text)

    posts = np.array(posts)
    labels = np.array(labels)

    # If we only want to load a certain percentage of class 2, then take out all the class 2
    # samples, choose randomly from them, and then put only those ones back.
    if fraction_class_2_to_load < 1.0:
        class_2_indexes = np.where(labels == 2)[0]
        everything_else = np.where(labels != 2)[0]
        class_2_subset_indexes = np.random.choice(
            class_2_indexes, size=class_2_max)

        class_2_posts = posts[class_2_subset_indexes]
        class_2_labels = labels[class_2_subset_indexes]

        all_other_posts = posts[everything_else]
        all_other_labels = labels[everything_else]

        posts = np.concatenate([all_other_posts, class_2_posts])
        labels = np.concatenate([all_other_labels, class_2_labels])

    return (posts, labels)


def create_new_rows(seed_rows, num_new_rows, new_rows, aug):
    """
    Helper method for data augmentation that creates new data rows by randomly choosing 
    a row from the given list of seed rows, augmenting the relevant text fields of that 
    row, and collecting these new rows until we have generated `num_new_rows` of samples.
    Takes in: 
    - seed_rows: a list of row tuples that correspond to the rows of the original CSV
    - num_new_rows: an integer
    - new_rows: the list that this method will directly add new samples to
    - aug: the data augmentation model to use to generate new samples 
    """

    # Storing these instance methods locally makes performance marginally faster.
    augment = aug.augment
    append = new_rows.append
    for i in range(num_new_rows):
        print(i)
        row = random.choice(seed_rows)
        # Augment the post title
        row[2] = augment(row[2])
        # Augment the post body
        row[3] = augment(row[3])
        append(row)


def augment_data(num_new_class_0, num_new_class_1, clear_old_augmented_data=False, write_to_path='data/synonym_augmented_reddit_submissions.csv'):
    """
    Generates augmented data by producing new samples for class 0 and/or class 1, the two 
    classes that are underrepresented in our dataset, and writing them to a designated new
    file 'data/augmented_reddit_submissions.csv'.
    Takes in:
    - num_new_class_0: Integer representing how many new samples of class 0 to generate
    - num_new_class_1: Integer representing how many new samples of class 1 to generate
    - clear_old_augmented_data: Boolean; if set to True, will overwrite the old augmented data rather than 
    - write_to_path: The path of the file to write or append the new samples to.

    This function makes use of the nlpaug library's word augmenter. 
    """

    # We experimented with a couple other nlpaug models, but we ended up choosing SynonymAug
    # because it gave us the most natural-sounding and least noisy samples.
    # Other models we tried were:
    #   naw.WordEmbsAug             this one uses word2vec to find similar words for augmentation; it
    #                               ended up giving us very noisy data that made the performance of
    #                               all models decrease.
    #   naw.ContextualWordEmbsAug   this one uses BERT to do the same as the above; it was slightly
    #                               better, but still pretty noisy.
    aug = naw.SynonymAug(aug_src='wordnet')

    new_rows = []
    with open('data/reddit_submissions.csv') as f:
        reader = csv.reader(f)
        # Skip the first row that just has column names
        rows = list(reader)[1:]
        print('unfiltered rows: {}'.format(len(rows)))

        seed_rows_with_class_0 = list(
            filter(lambda r: CLASSES[r[0]] == 0, rows))
        seed_rows_with_class_1 = list(
            filter(lambda r: CLASSES[r[0]] == 1, rows))
        print('filtered rows: {}'.format(
            len(seed_rows_with_class_0) + len(seed_rows_with_class_1)))

        print('generating new data with class 0')
        create_new_rows(seed_rows_with_class_0, num_new_class_0, new_rows, aug)
        print('generating new data with class 1')
        create_new_rows(seed_rows_with_class_1, num_new_class_1, new_rows, aug)

    file_open_mode = 'w' if clear_old_augmented_data else 'a'

    with open(write_to_path, file_open_mode) as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        print('writing new rows')
        writer.writerows(new_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--generate_augmented_data',
                        action='store_true', help='If set, then generates augmented data into data/augmented_reddit_submissions.csv when the script is run')
    parser.add_argument('--clear_old_augmented_data',
                        action='store_true', help='If set, then overwrites the old augmented data with new augmented data')
    parser.add_argument('--num_new_class_0',
                        type=int, default=1000, help='The number of samples of class 0 to generate')
    parser.add_argument('--num_new_class_1',
                        type=int, default=1000, help='The number of samples of class 1 to generate')
    parser.add_argument('--write_to_path',
                        type=str, default='data/synonym_augmented_reddit_submissions.csv',
                        help='The relative path of the file to write or append the new data to')

    args = parser.parse_args()

    if args.generate_augmented_data:
        nltk.download('wordnet')
        augment_data(
            num_new_class_0=args.num_new_class_0,
            num_new_class_1=args.num_new_class_1,
            clear_old_augmented_data=args.clear_old_augmented_data,
            write_to_path=args.write_to_path)
