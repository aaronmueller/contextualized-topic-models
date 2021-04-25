from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
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

def func(x, a, b, c):
  # return a * np.exp(-b * x) + c
  return a * np.log(b * x) + c

code_to_name = {
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'pt': 'Portuguese',
    'nl': 'Dutch'
}

sns.set()
sns.set(font_scale=1.5)
# sns.set_style('dark_grid')
index = ['CTM', 'CTM+FT (NLI)', 'CTM+FT (DC)', 'CTM+FT (TC, Patents)', \
    'CTM+FT (TC, Wiki)', 'CPT', 'CPT+FT (TC, Wiki)']
sts_scores = [40.73, 82.5, 44.41, 53.32, 51.37, 43.73, 48.4]
npmi_scores = [.144, .150, .156, .154, .16, .147, .151]
match_scores = [33.3, 55.33, 41.36, 51.11, 47.17, 33.1, 49.23]
kl_scores = [0.56, 0.35, 0.51, 0.37, 0.44, 0.55, 0.4]

fig, ax1 = plt.subplots()
ax1.set_xlabel('STS performance')
# ax1.grid(False)
# ax1.set_xscale('log')
ax1.set_ylabel('Match')
a = ax1.scatter(sts_scores, match_scores, color='#4477b2', label="Match")
ax1.yaxis.label.set_color('#4477b2')
ax1.tick_params(axis='y', colors='#4477b2')
#popt, pcov = curve_fit(func, sts_scores, match_scores)
#x = np.linspace(min(sts_scores),max(sts_scores))
#ax1.plot(x, func(x, *popt), linestyle='--', color='#779ecb')

ax2 = ax1.twinx()
# ax2.set_xscale('log')
ax2.set_ylabel('KL')
b = ax2.scatter(sts_scores, kl_scores, color='#d92121', label="KL", marker='x')
ax2.yaxis.label.set_color('#d92121')
ax2.grid(False)
ax2.tick_params(axis='y', colors='#d92121')
p = [a, b]
ax1.legend(p, [p_.get_label() for p_ in p], loc='center right')
# plt.legend()
#popt, pcov = curve_fit(func, sts_scores, kl_scores)
#x = np.linspace(min(sts_scores),max(sts_scores))
#ax2.plot(x, func(x, *popt), linestyle='--', color='#ff5148')

plt.savefig(os.path.join("scatterplots", "scatter_matchkl.pdf"), format="pdf", \
    bbox_inches="tight")

print("STS vs. NPMI:", spearmanr(sts_scores, npmi_scores))
print("STS vs. Match:", spearmanr(sts_scores, match_scores))
print("STS vs. KL:", spearmanr(sts_scores, kl_scores))

'''
df = pd.DataFrame(zip(sts_scores, npmi_scores, match_scores, kl_scores), \
    columns=['STS', 'NPMI', 'Match', 'KL'], index=index)
print(df.head())

out_path = 'scatterplots'
if not os.path.exists(out_path):
    os.makedirs(out_path)
ax = sns.scatterplot(data=df, x="STS", y="KL")
plt.savefig(os.path.join(out_path, "scatter_matchkl.pdf"), format="pdf", \
    bbox_inches='tight')
'''