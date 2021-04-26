# Fine-tuning Encoders for Improved Monolingual and Zero-shot Polylingual Neural Topic Modeling

This respository contains code for replicating the experiments of our NAACL 2021 paper, [Fine-tuning Encoders for Improved Monolingual and Zero-shot Polylingual Neural Topic Modeling](https://arxiv.org/abs/2104.05064). Specifically, this repository contains code for preprocessing input data, the article IDs for the Wikipedia dataset we use in the paper, and the code for TCCTM modeling. This repository is very similar to the original [contextualized topic modeling repository](https://github.com/MilaNLProc/contextualized-topic-models), but with the addition of our specific TCCTM model and evaluation code.

For continued pre-training, use the [huggingface transformers](https://github.com/huggingface/transformers) repository. We have included our continued pre-training script in the `cpt` folder.

For fine-tuning sentence embeddings, use the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) repository.

Contextualized Topic Models (CTMs) are a family of topic models that use pre-trained representations of language (e.g., BERT) to
support topic modeling. See the original CTM papers for details:

* Cross-lingual Contextualized Topic Models with Zero-shot Learning: https://arxiv.org/pdf/2004.07737v1.pdf
* Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence: https://arxiv.org/pdf/2004.03974.pdf


## Data

We use a subset of the aligned multilingual Wikipedia Comparable Corpora dataset, which may be found here: https://linguatools.org/tools/corpora/wikipedia-comparable-corpora/

For training, we use 100,000 aligned articles per-language; for testing, we hold out 10,000 aligned articles per-language. Because aligned articles are presented in the same XML object and use the same English title across language pairs, we use English article names as identifiers. You can find a (non-comprehensive) list of article titles for articles shared by all languages used in this study in `contextualized_topic_models/data/wiki/common_articles.txt`. In this study, we (arbitrarily) sampled the first 10,000 article titles from that list and obtained their associated article texts from the Comparable Corpora dataset to create our aligned test set. Similarly, we sampled the last 100,000 article titles and obtained those article texts to create our aligned training set.

We create aligned testing sets for cross-lingual evaluation. But if we only train on English, why create an aligned training set? This is to generate language-specific vocabularies that have (ideally 100%, but realistically a bit less) overlap in lexical-semantic content cross-linguistically.

## Preprocessing

We include our preprocessing notebook in `examples/preprocessing_wiki.ipynb`. Note that to use this, you will first need to generate vocabularies for each language. The vocabularies should be text files where each line contains one token.  We simply took the 5000 most frequent tokens per-language, though the original CTM paper used 2000 tokens per-language. There is also a built-in preprocessing script in the [original CTM repository](https://github.com/MilaNLProc/contextualized-topic-models).

## Training

To train a regular CTM, use the `model_wiki.py` script. This script is currently instantiated with the best hyperparameters we found on the dataset used in our paper. Note that you will need to modify the paths

To train a TCCTM, use the `model_wiki_topicreg.py` script. The primary difference between this and `model_wiki.py` is that this script uses a new `CTMDatasetTopReg` data processor, rather than the default `CTMDataset`; this data processor loads the input data as well as topic labels for each article. The document labels are generated from an LDA topic model. When the `CTMDatasetTopReg` processor is used, the TCCTM model is automatically used without any further changes needed in the main code. This behavior is defined in the `CTM` class of the [CTM model definition script](contextualized_topic_models/models/ctm.py).

![TCCTM architecture](img/tcctm_architecture.png)

The difference between a CTM model and TCCTM model is that the TCCTM contains a topic classifier. The model maps from the hidden representation of the input sentences produced by the VAE to a topic label, using a negative log-likelihood loss. This loss is added to the loss of the topic model. If you do not wish to fine-tune your contextualized sentence embeddings before applying them to monolingual topic modeling, TCCTM achieves similar performance to a CTM with well-tuned sentence embeddings for this task. However, note that if you want good zero-shot cross-lingual topic transfer, you will want to fine-tune your embeddings.

## Evaluation

To obtain NPMI coherence scores for a trained (TC)CTM, use the `topic_inference.py` script. The usage is as follows:
```
python topic_inference.py <model_file.pth> <epoch> 
```
By default, training happens for 60 epochs, and only the last epoch is saved. If you use this code as-is, use `59` (epochs are 0-indexed here) for the `<epoch>` argument.

To perform cross-lingual evaluations, use the `multiling_eval.py` script. The usage is as follows:
```
python multiling_eval.py <model_file.pth> <epoch> <sbert_model>
```
where `<sbert_model>` is the output directory of a trained `sentence-transformers` model. This script will output Match and KL scores for the aligned English-{French, German, Dutch, Portuguese} test sets.


## License & Documentation

As this repository is forked from a repository which uses the MIT License, we also use the MIT License. You may freely reuse code found here in proprietary software, provided you include the MIT License terms and copyright notice.

* Free software: MIT license
* Further CTM Documentation: https://contextualized-topic-models.readthedocs.io.

## References

If you use the materials in this repository in a research work, please cite this paper:

```
    @inproceedings{mueller2021finetuning,
        title={Fine-tuning Encoders for Improved Monolingual and Zero-shot Polylingual Topic Modeling},
        author={Aaron Mueller and Mark Dredze},
        year={2021},
        booktitle="Proceedings of the 2021 Conference of the North {A}merican Chapter of the Association for Computational Linguistics, Volume 1 (Long and Short Papers)",
        publisher="Association for Computational Linguistics"
    }
```

In addition, please cite the following papers on contextualized topic modeling:

```
    @article{bianchi2020pretraining,
        title={Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence},
        author={Federico Bianchi and Silvia Terragni and Dirk Hovy},
        year={2020},
       journal={arXiv preprint arXiv:2004.03974},
    }


    @article{bianchi2020crosslingual,
        title={Cross-lingual Contextualized Topic Models with Zero-shot Learning},
        author={Federico Bianchi and Silvia Terragni and Dirk Hovy and Debora Nozza and Elisabetta Fersini},
        year={2020},
       journal={arXiv preprint arXiv:2004.07737},
    }
```
