import csv
import numpy as np
import random

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

import argparse

# The domesticviolence and survivorsofabuse subreddits will be class 0, critical; these are personal stories, calls for help, requests for advice
# The abuseInterrupted subreddit will be class 1, noncritical; it mostly contains empty text, links to articles, general statements about abuse, etc.
# Everything else will be class 2, general/unrelated

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


def load_data(file_path, include_og=True, include_aug=False, fraction_class_2_to_load=1.0):
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


def augment_data(num_new_class_0, num_new_class_1, clear_old_augmented_data=False):
    # aug = naw.WordEmbsAug(
    # model_type='word2vec', model_path='./GoogleNews-vectors-negative300.bin',
    # action="substitute")
    # aug = naw.SynonymAug(aug_src='wordnet')
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert", device='cpu')

    new_rows = []
    with open('data/reddit_submissions.csv') as f:  # open with read permission
        reader = csv.reader(f)
        # Skip the first row that just has column names
        rows = list(reader)[1:]
        print('unfiltered rows: {}'.format(len(rows)))
        rows_without_class_2 = list(filter(lambda r: CLASSES[r[0]] != 2, rows))
        print('filtered rows: {}'.format(len(rows_without_class_2)))
        print('generating new data')
        # apparently saving these locally is faster
        augment = aug.augment
        for i in range(10):
            print(i)
            row = random.choice(rows_without_class_2)
            # Augment the post title
            row[2] = augment(row[2])
            # Augment the post body
            row[3] = augment(row[3])
            new_rows.append(row)

    with open('data/augmented_reddit_submissions.csv', 'a') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        print('writing new rows')
        writer.writerows(new_rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--generate_augmented_data',
        action='store_true', help='If set, then trains network')
    parser.add_argument('--clear_old_augmented_data',
        action='store_true', help='If set, then trains network')
    parser.add_argument('--num_new_class_0',
        type=int, default=1000, help='Base learning rate (alpha)')
    parser.add_argument('--num_new_class_1',
        type=int, default=1000, help='Base learning rate (alpha)')

    args = parser.parse_args()

    augment_data(
        num_new_class_0=args.num_new_class_0,
        num_new_class_1=args.num_new_class_1,
        clear_old_augmented_data=args.clear_old_augmented_data)

