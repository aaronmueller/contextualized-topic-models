import pickle
import torch
import sys
from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import TextHandler, bert_embeddings_from_file

NUM_TEST_TOKENS = 683563

def show_topics(topic_list):
    for idx, topic_tokens in enumerate(topic_list):
        print(idx)
        print(' '.join(topic_tokens))

if len(sys.argv) < 2:
    raise Exception("Usage: python {} {}".format(sys.argv[0], "<model_file>"))

handler = TextHandler("contextualized_topic_models/data/wiki/wiki_test_en_prep_sub.txt")
# handler = TextHandler("contextualized_topic_models/data/iqos/iqos_corpus_prep_en.txt")
handler.prepare()

ctm = CTM(input_size=len(handler.vocab), inference_type="contextual", bert_input_size=768)
ctm.load(sys.argv[1], sys.argv[2])

test_bert = bert_embeddings_from_file('contextualized_topic_models/data/wiki/wiki_test_en_unprep_sub.txt', \
        sys.argv[3])
testing_dataset = CTMDataset(handler.bow, test_bert, handler.idx2token)

# print(ctm.get_topic_lists(10))
# show_topics(ctm.get_topic_lists(10))
ctm.test(testing_dataset, NUM_TEST_TOKENS)