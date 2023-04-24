import os
import pickle
import numpy as np
import hnswlib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import util
from datasets import load_dataset
import evaluate
from models import NLI, PP_Detector
import spacy

def get_sim_texts(model, query, corpus_sentences_pair,
                  index, top_k=3):

    query_embedding = model.encode(query)

    #We use hnswlib knn_query method to find the top_k_hits
    corpus_ids, distances = index.knn_query(query_embedding, k=top_k)

    # We extract corpus ids and scores for the first query
    hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'])

    sel_corpus = [corpus_sentences_pair[hit['corpus_id']] for hit in hits[0:top_k]]

    return sel_corpus

def get_corpus_emb(embedding_cache_path, model, data_name='quora'):
    corpus_dataset = load_dataset(data_name, split="train")
    #Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        corpus_sentences = [t['text'][0] for t in corpus_dataset['questions']]
        print("Encode the corpus. This might take a while")
        corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

        print("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
    else:
        print("Load pre-computed embeddings from disc")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            corpus_sentences = cache_data['sentences']
            corpus_embeddings = cache_data['embeddings']
    return corpus_embeddings, corpus_dataset

def create_search_index(corpus_embeddings, embedding_size=768, index_path="./hnswlib.index"):
    #We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
    index = hnswlib.Index(space = 'cosine', dim = embedding_size)

    if os.path.exists(index_path):
        print("Loading index...")
        index.load_index(index_path)
    else:
        ### Create the HNSWLIB index
        print("Start creating HNSWLIB index")
        index.init_index(max_elements = len(corpus_embeddings), ef_construction = 400, M = 64)

        # Then we train the index to find a suitable clustering
        index.add_items(corpus_embeddings, list(range(len(corpus_embeddings))))

        print("Saving index to:", index_path)
        index.save_index(index_path)

    # Controlling the recall by setting ef:
    index.set_ef(50)  # ef should always be > top_k
    return index

def get_sim_pp(pp_detector, all_pps, target_sent, T=0.8):
    pps = []
    for sentence in all_pps:
        score = pp_detector.score_binary(
            target_sent,
            sentence
        )
        if score[1] >= T:
            pps.append(sentence)
    pps = set([pp.lower() for pp in pps])
    return pps


class ConsistencyScoring():
    def __init__(self, tok_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector",\
                 model_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector"):
        super(ConsistencyScoring, self).__init__()
        self.bleu = evaluate.load('bleu')
        self.bertscore = evaluate.load("bertscore")
        self.pp_detector = PP_Detector(tok_path, model_path)
        self.nli = NLI()
        self.NER = spacy.load("en_core_web_sm")

    def lexical_agreement(self, output_i, output_j):
        if not output_i:
            return 0
        if not output_j:
            return 0
        bleu_score = self.bleu.compute(predictions=[output_i], references=[output_j])
        return bleu_score['bleu'] or 0.0

    def bertscore_agreement(self, output_i, output_j):
        bertscore_score = self.bertscore.compute(predictions=[output_i], references=[output_j], lang='en')
        return bertscore_score['f1'][0]

    def pp_agreement(self, output_i, output_j):
        pp_detector_score = self.pp_detector.score_binary(output_i, output_j)
        return pp_detector_score[1]

    def entailment_agreement(self, output_i, output_j):
        return self.nli.entailed(output_i, output_j)

    def contradiction_agreement(self, output_i, output_j):
        return self.nli.contradicted(output_i, output_j)

    def consistency(self, outputs, agreement_fn, threshold, binary=True):
        agreements = 0
        for i, output_i in enumerate(outputs):
            for j, output_j in enumerate(outputs):
                if i == j:
                    continue
                agreement_score = agreement_fn(output_i, output_j)
                if binary and agreement_score >= threshold:
                    agreements += 1
                elif binary == False:
                    agreements += agreement_score
        if (len(outputs) * (len(outputs) - 1)) == 0:
            return 0
        return (1 / (len(outputs) * (len(outputs) - 1))) * agreements

    def ner_match_score(self, text_i, text_j):
        pro_texti = self.NER(text_i)
        pro_textj = self.NER(text_j)
        num_matches = 0
        all_NERs = []
        for word_i in pro_texti.ents:
            for word_j in pro_textj.ents:
                all_NERs.extend([word_i.text, word_j.text])
                if word_i.text == word_j.text:
                    num_matches += 1
                    break # no multiple match
        if len(all_NERs) == 0:
          return 0.0
        return float(num_matches/len(set(all_NERs)))

    def get_score(self, outputs):
        fns = [
            (self.lexical_agreement, 0.5),
             (self.bertscore_agreement, 0.9),
             (self.pp_agreement, 0.8),
            (self.entailment_agreement, 0.5),
            (self.contradiction_agreement, 0.5),
            (self.ner_match_score, 0.8)
        ]
        con_scores = {}
        for fn, threshold in fns:
            print('Getting score for ', fn.__name__+'_binary')
            con_scores[fn.__name__+'_binary'] = self.consistency(outputs, fn, threshold, True)

        for fn, threshold in fns:
            print('Getting score for ', fn.__name__)
            con_scores[fn.__name__] = self.consistency(outputs, fn, threshold, False)
        return con_scores
