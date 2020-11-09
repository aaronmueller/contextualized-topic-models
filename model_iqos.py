from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file, bert_embeddings_from_list
import os
import numpy as np
import pickle
import torch
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import TextHandler

handler = TextHandler("contextualized_topic_models/data/iqos/iqos_corpus_prep_en.txt")
handler.prepare()

# train_bert = bert_embeddings_from_file('contextualized_topic_models/data/iqos/iqos_corpus_unprep_en.txt', 'distiluse-base-multilingual-cased')
# train_bert = bert_embeddings_from_file('contextualized_topic_models/data/iqos/iqos_corpus_unprep_en.txt', 'xlm-r-100langs-bert-base-nli-mean-tokens')
train_bert = bert_embeddings_from_file('contextualized_topic_models/data/iqos/iqos_corpus_unprep_en.txt', \
        '/export/c04/amueller/sentence-transformers/sentence_transformers/output/training_topics_4_iqos-xlmr-2020-10-05_13-15-51')
training_dataset = CTMDataset(handler.bow, train_bert, handler.idx2token)

num_topics = 20
ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, num_epochs=60, hidden_sizes=(100,),
          inference_type="contextual", n_components=num_topics, num_data_loader_workers=0)
# ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, num_epochs=60, hidden_sizes=(100,),
#          inference_type="contextual", n_components=num_topics, num_data_loader_workers=0)
ctm.fit(training_dataset)

# filehandler = open("iqos_en.ctm", 'wb')
torch.save(ctm, "iqos_en_xlmr_topics_cpu_ct_4.ctm")
