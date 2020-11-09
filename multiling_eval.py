import torch
import sys
import numpy as np
from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.evaluation.measures import Matches, KLDivergence, CentroidDistance
from contextualized_topic_models.utils.data_preparation import TextHandler, bert_embeddings_from_file
from matplotlib import pyplot as plt
from collections import Counter

def show_topics(topic_list):
    for idx, topic_tokens in enumerate(topic_list):
        print(idx)
        print(' '.join(topic_tokens))

if len(sys.argv) < 4:
    raise Exception("Usage: python {} {} {} {}".format(sys.argv[0], "<ctm_model>", "<checkpoint>", "<sbert_model>"))

handler_en = TextHandler("contextualized_topic_models/data/wiki/wiki_test_en_prep_sub.txt")
# handler = TextHandler("contextualized_topic_models/data/wiki/iqos_corpus_prep_en.txt")
handler_en.prepare()

testing_bert_en = bert_embeddings_from_file("contextualized_topic_models/data/wiki/wiki_test_en_unprep_sub.txt", sys.argv[3])
testing_dataset_en = CTMDataset(handler_en.bow, testing_bert_en, handler_en.idx2token)

ctm = CTM(input_size=len(handler_en.vocab), inference_type="contextual", bert_input_size=768)
# ctm = torch.load(sys.argv[1], map_location="cpu")
ctm.load(sys.argv[1], sys.argv[2])

num_topics = 100
thetas_en = ctm.get_thetas(testing_dataset_en, n_samples=100)
with open("temp/topics_en.txt", 'w') as test_out:
    topics = np.squeeze(np.argmax(thetas_en, axis=1).T)
    for topic in topics:
        test_out.write('\n' + str(topic))
        
# plot topic histogram
labels, values = zip(*Counter(np.squeeze(np.argmax(thetas_en, axis=1).T)).items())
indexes = np.arange(len(labels))
width = 1
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.savefig('figures/hist_en.png') 
# print(thetas_en)

# uniform baseline
# thetas_en = np.random.uniform(size=(len(testing_dataset_en), num_topics))
# thetas_en = thetas_en / np.sum(thetas_en, axis=1)[:, np.newaxis]

scores = {'match': [], 'kl': []}

for lang in ('fr', 'de', 'nl', 'pt'):
    handler_fr = TextHandler(f"contextualized_topic_models/data/wiki/wiki_test_{lang}_prep_sub.txt")
    handler_fr.prepare()
    testing_bert_fr = bert_embeddings_from_file(f'contextualized_topic_models/data/wiki/wiki_test_{lang}_unprep_sub.txt', sys.argv[3])
    testing_dataset_fr = CTMDataset(handler_fr.bow, testing_bert_fr, handler_fr.idx2token)
    thetas_fr = ctm.get_thetas(testing_dataset_fr, n_samples=100)
    with open(f"temp/topics_{lang}.txt", 'w') as test_out:
        topics = np.squeeze(np.argmax(thetas_fr, axis=1).T)
        for topic in topics:
            test_out.write('\n' + str(topic))
    # plot topic histogram
    plt.cla(); plt.clf()
    labels, values = zip(*Counter(np.squeeze(np.argmax(thetas_fr, axis=1).T)).items())
    indexes = np.arange(len(labels))
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.savefig(f'figures/hist_{lang}.png') 
    # calculate multilingual metrics
    match = Matches(thetas_en, thetas_fr)
    kl = KLDivergence(thetas_en, thetas_fr)
    # centroid = CentroidDistance(thetas_en, thetas_fr, num_topics)
    print('{} results:'.format(lang))
    print('\tmatch: {}\tkl: {}'.format(match.score(), kl.score()))
    scores['match'].append(match.score())
    scores['kl'].append(kl.score())
    # scores['centroid'].append(centroid)

print("Matches:", np.mean(scores['match']))
print("KL Divergence:", np.mean(scores['kl']))
# print("Centroid Distance:", np.mean(scores['centroid']))