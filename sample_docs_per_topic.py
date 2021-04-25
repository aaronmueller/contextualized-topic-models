from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import defaultdict
import os
import sys
import random
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def load_topics(topic_file):
    topic_list = []
    with open(topic_file, 'r') as topics:
        for topic in topics:
            topic_list.append(int(topic.strip()))
    return topic_list

code_to_name = {
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'pt': 'Portuguese',
    'nl': 'Dutch'
}
en_topics = np.array(load_topics("temp/topics_en.txt"))
lang_topics = {}
for lang in ('fr', 'de', 'pt', 'nl'):
    lang_topics[lang] = np.array(load_topics(f"temp/topics_{lang}.txt"))

topic_of_interest = int(sys.argv[1])
# topic_indices = set(np.where(en_topics != 81)[0])
topic_indices = set(np.where(en_topics == topic_of_interest)[0])
#for lang in ('fr', 'de', 'pt', 'nl'):  
topic_indices = topic_indices.intersection(set(np.where(lang_topics['fr'] == topic_of_interest)[0]))
# print(lang_topics['fr'][np.where(en_topics == topic_of_interest)])
# topic_indices = random.sample(topic_indices, 5)

with open("contextualized_topic_models/data/wiki/wiki_test_nl_unprep_sub.txt", 'r') as docs:
    for idx, doc in enumerate(docs):
        if idx in topic_indices:
            print(idx, lang_topics['nl'][idx], doc)