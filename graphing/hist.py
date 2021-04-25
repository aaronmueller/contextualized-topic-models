from sklearn.metrics import confusion_matrix
from collections import Counter
import os
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
en_topics = load_topics("temp/topics_en.txt")
non_num = -1
for num in range(100):
    if num not in np.unique(en_topics):
        en_topics.append(num)
        non_num = num

# df_hist = pd.DataFrame(en_topics, columns=['Topic'])
sns.set()
sns.set(font_scale=2.0)
# print(len(df_hist['Topic']))
# cmap = sns.color_palette("Reds", 256)
# cmap = sns.color_palette("Blues", as_cmap=True)
labels, values = zip(*Counter(en_topics).items())
indexes = np.arange(len(labels))
width = 1.0

f, ax = plt.subplots(figsize=(10,10))
ax.bar(indexes, values, width)
ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 100, 10)))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("English Topic")
ax.set_ylabel("Count")
ax.set_ylim([0,250])
# plt.xticks(indexes + width * 0.5, labels)

out_path = "hist_plots"
if not os.path.exists(out_path):
    os.makedirs(out_path)
plt.savefig(os.path.join(out_path, "hist_en_big.pdf"), format="pdf", \
    bbox_inches='tight')

plt.cla(); plt.clf()

for lang in ('fr', 'de', 'pt', 'nl'):
    lang_name = code_to_name[lang]
    lang_topics = load_topics(f"temp/topics_{lang}.txt")
    lang_topics.append(non_num)
    labels, values = zip(*Counter(lang_topics).items())
    indexes = np.arange(len(labels))
    width = 1.0

    f, ax = plt.subplots(figsize=(10,10))
    ax.bar(indexes, values, width)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(f"{lang_name} Topic")
    ax.set_ylabel("Count")
    plt.savefig(os.path.join(out_path, f"hist_{lang}_big.pdf"), format="pdf", \
        bbox_inches='tight')