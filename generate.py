import argparse
import os
import json
import requests
from datetime import timedelta, datetime, timezone

import pandas as pd
from tqdm import tqdm

import torch
import torch.distributed as dist

from src.utils import *
from src.models import *


def generate(bench:pd.DataFrame, model:base_model, verbose:bool) -> pd.DataFrame:
    '''
    Generates results for one process.

    Given the benchmark data file or a subset of it in `bench`, generates the
    output DataFrame containing the model's responses to each question.
    '''
    result = bench.drop('image', axis=1)
    preds = []
    with torch.inference_mode():
        for _, row in tqdm(bench.iterrows(), total=len(bench.index), smoothing=0.1):
            img = img_decode(row['image'])
            pred = model.generate(img, row['question'])
            if verbose: print(pred, flush=True)
            preds.append(pred)
    result['prediction'] = preds
    return result


def gen_meta_info(vlm:base_model, modelpath:str, adapterpath:str, **kwargs):
    '''
    Generates a dictionary containing meta information for the evaluation, to be
    saved as a JSON file in the same directory as the output.
    '''
    # number of parameters in the model
    if isinstance(vlm, api_model):
        num_params = None
    else:
        try: num_params = vlm.model.num_parameters()
        except AttributeError: num_params = sum(p.numel() for p in vlm.model.parameters())

    # date of evaluation
    eval_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    name = modelpath
    if adapterpath != '': name = adapterpath

    # model name should be clickable in the leaderboard if it is a huggingface repo
    clickable = False
    if not os.path.exists(name):
        try:
            if repo_exists(name):
                clickable = True
        except HFValidationError:
            pass

    name = os.path.basename(name)

    return {
        'meta_version': '0',
        'name': name,
        'modelpath': modelpath,
        'adapterpath': adapterpath,
        'gen_kwargs': vlm.gen_kwargs,
        'eval_date': eval_date,
        'num_params': num_params,
        'exclude': kwargs.get('exclude', False), # exclude from leaderboard
        'clickable': clickable,
    }


def main(data:str, modelpath:str, adapterpath:str, outfile:str, verbose:bool):
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(seconds=10800))

    if not os.path.exists(data):
        if data != 'data/KOFFVQA.tsv':
            raise FileNotFoundError('Data file does not exist.')
        if rank == 0:
            print('data/KOFFVQA.tsv not found; downloading...')
            os.makedirs('data/', exist_ok=True)
            resp = requests.get('https://huggingface.co/datasets/maum-ai/KOFFVQA_Data/resolve/main/data/KOFFVQA.tsv')
            if not resp.ok:
                raise RuntimeError('File failed to download')
            with open(data, 'wb') as f:
                f.write(resp.content)

    if world_size > 1:
        dist.barrier()

    if rank == 0: print('loading model...', flush=True)
    model = auto_model(modelpath, adapterpath, auto_device=(world_size==1))

    meta_info = {}
    if rank == 0:
        meta_info = gen_meta_info(model, modelpath, adapterpath)

    if isinstance(model, api_model) and world_size > 1 and rank == 0:
        print('Warning: You are using an API model with Torch multiprocessing. This is unnecessary, and may also result in errors due to a high request rate.')

    if rank == 0: print('loading data...', flush=True)

    if world_size > 1:
        bench_ = [pd.read_csv(data, sep='\t') if rank == 0 else None]
        dist.barrier()
        dist.broadcast_object_list(bench_, 0)
        bench = bench_[0].iloc[rank::world_size] # split benchmark into subset between processes

        if rank == 0: print('starting predictions...', flush=True)

        result = generate(bench, model, verbose)
        dist.barrier()

        if rank == 0: print('saving...', flush=True)

        results = []
        for i in range(world_size):
            results.append(result if rank==i else None)
            dist.barrier()
            dist.broadcast_object_list(results, i)

        if rank == 0: # reassemble generated responses into one dataframe
            all_result = pd.concat(results, ignore_index=True).set_index('index')
            all_result = all_result.reindex(bench_[0].set_index('index').index).reset_index()

            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            all_result.to_excel(outfile, index=False, engine='xlsxwriter')

            with open(os.path.join(os.path.dirname(outfile), 'gen_meta_info.json'), 'w') as j:
                json.dump(meta_info, j, ensure_ascii=False, indent=2)

    else:
        bench = pd.read_csv(data, sep='\t')

        print('starting predictions...', flush=True)
        result = generate(bench, model, verbose)

        print('saving...', flush=True)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        result.to_excel(outfile, index=False, engine='xlsxwriter')

        with open(os.path.join(os.path.dirname(outfile), 'gen_meta_info.json'), 'w') as j:
            json.dump(meta_info, j, ensure_ascii=False, indent=2)

    if rank == 0: print('done.', flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser('KOFFVQA evaluation')
    parser.add_argument('--data', '-d', default='data/KOFFVQA.tsv', help='Path to the TSV file containing the benchmark. `data/KOFFVQA.tsv` by default.')
    parser.add_argument('--model', '-m', help='Path to the directory containing the model in Huggingface format, a Huggingface model name, or the name of a supported API model.')
    parser.add_argument('--adapter', '-a', default='', help='Path to the directory containing an adapter for the model in Huggingface format, if exists')
    parser.add_argument('--outfile', '-o', default='', help='Path to the file where the generation results will be saved. `result/{model name}_{adapter name}/{benchmark name}_gen.xlsx` by default.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Flag for verbose printing.')
    args = parser.parse_args()
    if args.outfile == '':
        args.outfile = f'result/{os.path.basename(args.model)}{"_" + os.path.basename(args.adapter) if args.adapter!="" else ""}/{os.path.basename(args.data).split(".")[0]}_gen.xlsx'
        if args.verbose and int(os.environ.get('RANK', 0)) == 0:
            print(f'Output file path not provided; defaulting to "{args.outfile}".', flush=True)

    main(args.data, args.model, args.adapter, args.outfile, args.verbose)
