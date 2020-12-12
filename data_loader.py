import csv
import numpy as np
import random

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import argparse

# only need to run once for synonym generator
# note: don't need this unless we augment again and it throws errors
# import nltk
# nltk.download('wordnet')

'''
The domesticviolence and survivorsofabuse subreddits will be class 0, critical; these are personal stories, calls for help, requests for advice
The abuseInterrupted subreddit will be class 1, noncritical; it mostly contains empty text, links to articles, general statements about abuse, etc.
Everything else will be class 2, general/unrelated
'''

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


def load_data(og_file_path, aug_file_path=None, include_og=True, include_aug=False, fraction_class_2_to_load=1.0, combine_classes_01=False):
    """
    Loads text and labels from dataset stored in file_path, a CSV.
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
                # print("\n" + str(row))
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

    if fraction_class_2_to_load < 1.0:
        class_2_indexes = np.where(labels == 2)[0]
        everything_else = np.where(labels != 2)[0]
        class_2_subset_indexes = np.random.choice(class_2_indexes, size=class_2_max)
        
        class_2_posts = posts[class_2_subset_indexes]
        class_2_labels = labels[class_2_subset_indexes]

        all_other_posts = posts[everything_else]
        all_other_labels = labels[everything_else]

        posts = np.concatenate([all_other_posts, class_2_posts])
        labels = np.concatenate([class_2_labels, all_other_labels])

    return (posts, labels)


def create_new_rows(seed_rows, num_new_rows, new_rows, aug):
    # apparently faster to store these locally, be careful with maintenance though
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


def augment_data(num_new_class_0=1, num_new_class_1=1, clear_old_augmented_data=False):
    # different models that can be used for augmenting:
    # aug = naw.WordEmbsAug(
    # model_type='word2vec', model_path='./GoogleNews-vectors-negative300.bin',
    # action="substitute")
    # aug = naw.ContextualWordEmbsAug(
    #     model_path='bert-base-uncased', action="insert", device='cpu')
    
    aug = naw.SynonymAug(aug_src='wordnet')

    new_rows = []
    with open('data/reddit_submissions.csv') as f:
        reader = csv.reader(f)
        # Skip the first row that just has column names
        rows = list(reader)[1:]
        print('unfiltered rows: {}'.format(len(rows)))
        seed_rows_with_class_0 = list(filter(lambda r: CLASSES[r[0]] == 0, rows))
        seed_rows_with_class_1 = list(filter(lambda r: CLASSES[r[0]] == 1, rows))
        print('filtered rows: {}'.format(len(seed_rows_with_class_0) + len(seed_rows_with_class_1)))
        print('generating new data with class 0')
        create_new_rows(seed_rows_with_class_0, num_new_class_0, new_rows, aug)
        print('generating new data with class 1')
        create_new_rows(seed_rows_with_class_1, num_new_class_1, new_rows, aug)

    with open('data/augmented_reddit_submissions.csv', 'a') as f:
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

    args = parser.parse_args()

    if args.generate_augmented_data:
        augment_data(
            num_new_class_0=args.num_new_class_0,
            num_new_class_1=args.num_new_class_1,
            clear_old_augmented_data=args.clear_old_augmented_data)

    else: 
        print("Testing data loader")

        posts, labels = load_data(og_file_path='data/reddit_submissions.csv')
        print('just og data:', len(posts), len(labels))
        class0 = labels[np.where(labels == 0)]
        print('just og data class 0:', len(class0))
        class1 = labels[np.where(labels == 1)]
        print('just og data class 1:', len(class1))
        class2 = labels[np.where(labels == 2)]
        print('just og data class 2:', len(class2))

        posts, labels = load_data(
            og_file_path='data/reddit_submissions.csv', 
            aug_file_path='data/augmented_reddit_submissions.csv', 
            include_og=True, 
            include_aug=True)
        print('og + aug data:', len(posts), len(labels))
        class0 = labels[np.where(labels == 0)]
        print('og + aug data class 0:', len(class0))
        class1 = labels[np.where(labels == 1)]
        print('og + aug data class 1:', len(class1))
        class2 = labels[np.where(labels == 2)]
        print('og + aug data class 2:', len(class2))

        posts, labels = load_data(
            og_file_path='data/reddit_submissions.csv', 
            aug_file_path='data/augmented_reddit_submissions.csv', 
            include_og=True, 
            include_aug=True,
            fraction_class_2_to_load=0.1)
        print('og + aug data with less 2:', len(posts), len(labels))
        class0 = labels[np.where(labels == 0)]
        print('og + aug data class 0 with less 2:', len(class0))
        class1 = labels[np.where(labels == 1)]
        print('og + aug data class 1 with less 2:', len(class1))
        class2 = labels[np.where(labels == 2)]
        print('og + aug data class 2 with less 2:', len(class2))
