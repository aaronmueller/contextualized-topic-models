import json
import sys
import pickle
from collections import defaultdict

with open("topic_full.json", "w") as train_file, open("wiki_train_en_prep.txt", "r") as text_prep, \
        open("wiki_train_en_unprep.txt", "r") as text_unprep:
    topic_list = pickle.load(open("english_topics_{}.pkl".format(sys.argv[1]), "rb"))
    topic_sets = {}
    for idx, topic in enumerate(topic_list):
        print(idx, end=" ")
        topic_sets[idx] = set()
        for word_and_prob in topic[1]:
            word = word_and_prob[0]
            topic_sets[idx].update([word])
        # print(' '.join(topic_sets[idx]))
    
    overall_topic_counts = defaultdict(int)
    text = []
    labels = []
    for line_prep, line_unprep in zip(text_prep, text_unprep):
        has_topic = False
        topic_word_counts = defaultdict(int)
        for token in line_prep.strip().split():
            for topic in topic_sets.keys():
                if token in topic_sets[topic]:
                    has_topic = True
                    topic_word_counts[topic] += 1

        # create new label for documents without tokens in any topic
        if not has_topic:
            text.append(line_unprep)
            labels.append(101)
            continue
        # classify document with topic it shares most tokens with
        max_topic = max(topic_word_counts, key=lambda key: topic_word_counts[key])
        overall_topic_counts[max_topic] += 1
        text.append(line_unprep)
        labels.append(max_topic)
    
    print(overall_topic_counts)
    assert len(text) == len(labels)
    json_list = []
    for example, label in zip(text, labels):
        json_list.append(json.dumps({'text': example, 'label': label}))

    train_file.write('\n'.join(json_list))
