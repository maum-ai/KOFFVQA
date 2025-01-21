import argparse
import os
import datetime
from string import punctuation, whitespace

import pandas as pd
from tqdm import tqdm
import py3langid as langid

import torch
import torch.distributed as dist

from src.utils import *
from src.judges import *


def evaluate(bench:pd.DataFrame, judge:base_judge, verbose:bool) -> pd.DataFrame:
    '''
    Evaluates results for one process.

    Given the generated predictions file or a subset of it in `bench`, generates
    the output DataFrame containing the judgement and scores.

    Four new columns are added:
    * `review`: The contents of the judge model's response.
    * `judge_score`: The score (0 to 10) parsed from the judgement.
    * `lang_filter`: A boolean value of whether the answer is in Korean.
    * `score`: The final score, which is either the `judge_score` value or zero
    depending on whether the answer is in Korean.
    '''
    result = bench.copy()
    preds = []
    scores = []
    lang_filters = []
    final_scores = []
    with torch.inference_mode():
        for _, row in tqdm(bench.iterrows(), total=len(bench.index), smoothing=0.1):
            # generate judgement
            judge_prompt = base_judge_prompt.format(row['question'], row['prediction'], row['answer'])
            pred = judge.generate(judge_prompt)
            if verbose: print(pred, flush=True)
            preds.append(pred)
            score = parse_score(pred)
            scores.append(score)

            # filter non-korean answers
            lang_code, _ = langid.classify(str(row['prediction']))
            # answers that are only numbers and punctuation should be counted even if they are not classified as korean
            only_numbers_and_punctuation = all(c.isdigit() or c in punctuation or c in whitespace for c in str(row['prediction']))
            if lang_code == 'ko' or only_numbers_and_punctuation:
                lang_filters.append(True)
                final_scores.append(score)
            else:
                lang_filters.append(False)
                final_scores.append(0)

    result['review'] = preds
    result['judge_score'] = scores
    result['lang_filter'] = lang_filters
    result['score'] = final_scores
    return result


def score(judgement:pd.DataFrame) -> pd.DataFrame:
    '''
    Given the constructed judgements file, creates a score summary.
    '''
    cates = ['overall'] + sorted(list(set(judgement['category'])))
    ret = {'split': [], 'score': []}
    for c in cates:
        ret['split'].append(c)
        sub = judgement[judgement['category'] == c] if c != 'overall' else judgement
        ret['score'].append(np.mean(sub['score']) * 10)
    return pd.DataFrame(ret)


def main(predfile:str, outfile:str, scorefile:str, judge:str, verbose:bool):
    assert os.path.exists(predfile), 'Predictions file does not exist'

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if rank == 0: print('loading judge...', flush=True)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))

    judge = auto_judge(judge, auto_device=(world_size==1))

    if rank == 0: print('loading data...', flush=True)

    if world_size > 1:
        bench_ = [pd.read_excel(predfile) if rank == 0 else None]
        dist.barrier()
        dist.broadcast_object_list(bench_, 0)
        bench = bench_[0].iloc[rank::world_size]

        if rank == 0: print('starting evaluations...', flush=True)

        result = evaluate(bench, judge, verbose)
        dist.barrier()

        if rank == 0: print('saving...', flush=True)

        results = []
        for i in range(world_size):
            results.append(result if rank==i else None)
            dist.barrier()
            dist.broadcast_object_list(results, i)

        if rank == 0:
            all_result = pd.concat(results, ignore_index=True).set_index('index')
            all_result = all_result.reindex(bench_[0].set_index('index').index).reset_index()

            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            all_result.to_excel(outfile, index=False, engine='xlsxwriter')
            score(all_result).to_csv(scorefile, index=False)

    else:
        bench = pd.read_excel(predfile)

        print('starting evaluations...', flush=True)
        result = evaluate(bench, judge, verbose)

        print('saving...', flush=True)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        result.to_excel(outfile, index=False, engine='xlsxwriter')
        score(result).to_csv(scorefile, index=False)

    if rank == 0: print('done.', flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser('KOFFVQA evaluation')

    parser.add_argument('--data', '-d', default='data/KOFFVQA.tsv', help='Path to the TSV file containing the benchmark. `data/KOFFVQA.tsv` by default.')
    parser.add_argument('--model', '-m', default='', help='Path to the directory containing the model in Huggingface format, or a Huggingface model name')
    parser.add_argument('--adapter', '-a', default='', help='Path to the directory containing an adapter for the model in Huggingface format, if exists')

    parser.add_argument('--predfile', '-i', default='', help='Path to the file containing generated predictions to evaulate. If provided, will override data and model arguments.')
    parser.add_argument('--outfile', '-o', default='', help='Path to the file where the evaluation results will be saved. `(prediction file path minus "_gen"}_eval.xlsx` by default.')
    parser.add_argument('--scorefile', '-s', default='', help='Path to the file where the evaluation scores will be saved. `(prediction file path minus "_gen"}_scores.csv` by default.')

    parser.add_argument('--judge', '-j', default='google/gemma-2-9b-it', help='Name or path of judge LLM. "google/gemma-2-9b-it" by default.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Flag for verbose printing.')

    args = parser.parse_args()

    rank = int(os.environ.get('RANK', 0))

    if args.predfile == '':
        args.predfile = f'result/{os.path.basename(args.model)}{"_" + os.path.basename(args.adapter) if args.adapter!="" else ""}/{os.path.basename(args.data).split(".")[0]}_gen.xlsx'
        if rank==0 and args.verbose: print(f'Predictions file path not provided; defaulting to "{args.predfile}".', flush=True)
    predfile_name = args.predfile.replace('.xlsx', '').replace('_gen', '')

    if args.outfile == '':
        args.outfile = f'{predfile_name}_eval.xlsx'
        if rank==0 and args.verbose: print(f'Output file path not provided; defaulting to "{args.outfile}".', flush=True)
    if args.scorefile == '':
        args.scorefile = f'{predfile_name}_scores.csv'
        if rank==0 and args.verbose: print(f'Scores file path not provided; defaulting to "{args.scorefile}".', flush=True)

    main(args.predfile, args.outfile, args.scorefile, args.judge, args.verbose)

