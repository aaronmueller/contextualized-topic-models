from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import defaultdict
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
print(non_num)


precisions = defaultdict(list)
recalls = defaultdict(list)
f1s = defaultdict(list)
for lang in ('fr','pt','de','nl'):
    lang_topics = load_topics(f"temp/topics_{lang}.txt")
    lang_topics.append(non_num)
    ps, rs, fs, supports = precision_recall_fscore_support(en_topics, lang_topics)
    precisions[lang].append(ps)
    recalls[lang].append(rs)
    f1s[lang].append(fs)

print("Highest prec:", np.argsort(np.mean([prec for prec in precisions.values()], axis=1))[0][-10:][::-1])
print("Lowest prec:", np.argsort(np.mean([prec for prec in precisions.values()], axis=1))[0][:5])
print("Highest recall:", np.argsort(np.mean([rec for rec in recalls.values()], axis=1))[0][-6:][::-1])
print("Lowest recall:", np.argsort(np.mean([rec for rec in recalls.values()], axis=1))[0][:5])
print("Highest f1:", np.argsort(np.mean([f1 for f1 in f1s.values()], axis=1))[0][-6:][::-1])
print("Lowest f1:", np.argsort(np.mean([f1 for f1 in f1s.values()], axis=1))[0][:5])

'''
# cmap = sns.color_palette("Reds", 256)
cmap = sns.color_palette("Blues", 256)

out_path = "confmat_figures"
if not os.path.exists(out_path):
    os.makedirs(out_path)

#sns.set_context("paper", rc={"font.size":18,"axes.labelsize":18})
sns.set(font_scale = 1.25)

for lang in ('fr', 'de', 'pt', 'nl'):
    lang_name = code_to_name[lang]
    lang_topics = load_topics(f"temp/topics_{lang}_simple.txt")
    #lang_topics.append(non_num)
    data = confusion_matrix(en_topics, lang_topics)
    df_cm = pd.DataFrame(data, columns=np.unique(en_topics), index=np.unique(en_topics))
    df_cm.index.name = "English Topic"
    df_cm.columns.name = f"{lang_name} Topic"

    # non-normalized figure
    ax = sns.heatmap(df_cm, cmap=cmap, annot=False)
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 100, 10)))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 100, 10)))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    plt.savefig(os.path.join(out_path, f"conf_mat_{lang}_simple_big.pdf"), format="pdf", \
        bbox_inches='tight', pad_inches=0.01)
    plt.cla(); plt.clf()

    # normalized figure
    df_cmn = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]
    df_cmn.index.name = "English Topic"
    df_cmn.columns.name = f"{lang_name} Topic"
    ax = sns.heatmap(df_cmn, cmap=cmap, annot=False, vmax=1.0)
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 100, 10)))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 100, 10)))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(left=True)
    plt.savefig(os.path.join(out_path, f"conf_mat_{lang}_norm_simple_big.pdf"), format="pdf", \
        bbox_inches='tight', pad_inches=0.01)
    plt.cla(); plt.clf()
'''