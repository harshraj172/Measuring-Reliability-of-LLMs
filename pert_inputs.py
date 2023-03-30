from tqdm.auto import tqdm
import argparse
import random
import pandas as pd
import torch
from models import *
from utils import *
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def evaluate(data):
    text = data['question']
    # paraphrase list starts with orig text
    pp_model_name = []

    ## Generate Paraphrases via different methods
    # Prompt LLMs
    exemplars_pair = get_sim_texts(model=embedder_txt,
                              query=text,
                              corpus_sentences_pair=[(t['text'][0], t['text'][1]) for t in corpus_dataset['questions']],
                              index=index,
                              top_k=args.num_exemplars)

    llm_pps, temperature, top_p = [], [0.0, 0.5, 1.0],[0.5, 1.0, 0.0]
    for j in range(3):
        llm_pps.extend(pp_llm_generator.generate_pp(exemplars_pair, text, args.num_pp, temperature[j], top_p[j]))
    pp_model_name.extend([f'Prompt-{args.gpt3_engine}']*len(llm_pps))

    # Finetuned T5
    cond_pps = pp_cond_generator.generate_pp(text, args.num_pp, sample=False)
    pp_model_name.extend(['T5-finetuned']*len(cond_pps))

    # Quality Control Generation
    qc_pps = []
    for _ in range(args.num_pp):
        lexical, syntactic, semantic = random.uniform(0.1, 0.8), random.uniform(0.1, 0.8), random.uniform(0.4, 1.0)
        qc_pps.append(pp_qc_generator.generate_pp(text, lexical=lexical, syntactic=syntactic, semantic=semantic)[0]['generated_text'])
    pp_model_name.extend(['QC']*len(qc_pps))

    return [data['dataset']]*len(pp_model_name), [data['passage']]*len(pp_model_name),\
         [data['question']]*len(pp_model_name), (llm_pps+cond_pps+qc_pps), pp_model_name

    # HRQVAE Model
    # pps.extend([pp[0] for pp in pp_hrqvae_generator.generate_pp([text])])
    # # Separator Model
    # print("Getting similar exemplars")
    # exemplars = get_sim_texts(query=list(text),
    #                             corpus=exemplars,
    #                             embedder=embedder_txt,
    #                             topk=args.num_reftext)
    # inp_text = [{"input": text, "exemplar": e} for e in exemplars]
    # pps.extend(pp_sep_generator.generate_pp(inp_text))

    # # Selects valid Paraphrases
    # pps = get_sim_pp(pp_detector=pp_detector,
    #                  all_pps=pps,
    #                  target_sent=text,
    #                  T=args.pp_thres,)


    # print("---------------------------------------------")
    # print(f"Paraphrases : {pps}")

    # gen_texts = llm.generate([{
    #             'passage': data['passage'],
    #             'question': pp
    #         } for pp in pps])

    # print(f"Generated Texts : {gen_texts}")

    # score = consistency_scoring([sentence]*len(texts))

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='parser to run the script')

    # add arguments
    parser.add_argument('--data_path',
                        type=str,
                        default='data/data-1.csv',
                        help='path to data in .csv')
    parser.add_argument('--gpt3_engine',
                        type=str,
                        choices=["text-davinci-002", "text-ada-001"],
                        default="text-ada-001",
                        help='gpt3 model to use, try using ada while testing')
    parser.add_argument('--exemplar_data_name',
                        type=str,
                        default='quora',
                        help='data to use selecting few shot examples')
    parser.add_argument('--num_exemplars',
                        type=int,
                        default=8,
                        help='no. of examples for few shot learning')
    parser.add_argument('--num_pp',
                        type=int,
                        default=6,
                        help='no. of paraphrases to generate from each method')
    parser.add_argument('--pp_thres',
                        type=float,
                        default=0.6,
                        help='similarity threshold above which a text is a paraphrase')

    args = parser.parse_args()

    # Selects true paraphrases
    # pp_detector = PP_Detector(tok_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector",\
    #                           model_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector", max_len=30)

    # Paraphrasers
    pp_cond_generator = Cond_PP(tok_path="doc2query/all-with_prefix-t5-base-v1", model_path="doc2query/all-with_prefix-t5-base-v1", template="question2question: {}")
    pp_qc_generator = QualityControl_PP('sentences')
    pp_llm_generator = LLM_PP(engine=args.gpt3_engine)

    embedder_txt = SentenceTransformer('paraphrase-mpnet-base-v2') # descibed in Novelty Controlled (RAPT) paper
    embedding_cache_path = '{}-embeddings-{}.pkl'.format(args.exemplar_data_name, 'paraphrase-mpnet-base-v2'.replace('/', '_'))
    corpus_embeddings, corpus_dataset = get_corpus_emb(embedding_cache_path=embedding_cache_path,\
                                                       model=embedder_txt, data_name=args.exemplar_data_name)
    index = create_search_index(corpus_embeddings, embedding_size=768, index_path="./hnswlib.index")

    df = pd.read_csv(args.data_path)

    data_names, passages, origs, pps, pp_model_names, k = [], [], [], [], [], 0
    print("Generating Input Perturbations/Paraphrases...")
    for id in tqdm(range(len((df)))):
        data_name, passage, orig, pp, pp_model_name = evaluate(df.iloc[id])
        data_names.extend(data_name)
        passages.extend(passage)
        origs.extend(orig)
        pps.extend(pp)
        pp_model_names.extend(pp_model_name)
        k += 1

        # to save
        if (k+1)%100==0:
            new_df = pd.DataFrame({ 'dataset': data_names,
                                'passage': passages,
                                'original question': origs,
                                'paraphrased question': pps,
                                'paraphrased generaton model': pp_model_names})
            new_df.to_csv('data/data-2.csv', index=False)