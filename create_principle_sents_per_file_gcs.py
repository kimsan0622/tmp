import os
import gcsfs
import json
import copy
from glob import glob
import argparse
import logging
from tqdm import tqdm
import functools
from multiprocessing import Pool, cpu_count

import numpy as np
from rouge_score.tokenizers import DefaultTokenizer


from rouge_scorer import RougeScorer
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="sent_eq_4k_25/en_nd1k_ml10k/*.jsonl",
        help="The input pattern for input files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='sent_score/en_nd1k_ml10k',
        help="path to output directory.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default='data-preproc',
        help="The project name of GCP.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default='preproc-bucket-1',
        help="The name of bucket",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=128,
        help="number of processes for preprocessing.",
    )
    parser.add_argument(
        "--rouge_types",
        nargs='+',
        type=str,
        default=['rouge1'],
        help="rouge types [rouge1, rouge2, rougeL]. e.g. --rouge_types rouge1 rouge2",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="start index for files",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="end index for files",
    )
    parser.add_argument(
        "--gsr",
        type=float,
        default=0.25,
        help="gap sentence ratio",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="name of tokenizer.",
    )

    args = parser.parse_args()
    return args


class PrincipleSentenceCreator:
    def __init__(self, 
            rouge_types,
            tokenizer=None, 
            use_stemmer=False) -> None:
            

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = DefaultTokenizer(use_stemmer)

        self.scorer = RougeScorer(
                rouge_types, 
                use_stemmer=False, 
                tokenizer=self._tokenizer
            )
    
    def get_f1_score(self, scores):
        fmeasure = [score.fmeasure for score in scores.values()]
        return sum(fmeasure)/len(fmeasure)

    def get_score(self, target, prediction, is_uniq=False):
        scores = self.scorer.score(
                target,
                prediction,
                is_uniq
            )
        return self.get_f1_score(scores)
    
    def get_score_pretokenized(self, target, prediction, is_uniq=False):
        scores = self.scorer.score_pretokenized(
                target,
                prediction,
                is_uniq
            )
        return self.get_f1_score(scores)
        

    def get_ps_ind(self, sents, m, is_uniq=False):
        len_sents = len(sents)
        mask = [0] * len_sents
        sents_tokens = [self._tokenizer.tokenize(sent) for sent in sents]

        for _ in range(m):
            scores = []
            for sent_idx in range(len_sents):
                if mask[sent_idx] == 1:
                    scores.append(0)
                    continue
            
                score = self.get_sent_score(sent_idx, mask, sents_tokens, is_uniq)
                scores.append(score)
            max_idx = np.argmax(scores)
            mask[max_idx] = 1
        return mask
    
    # greed search
    def get_ps_seq(self, sents, m, is_uniq=False):
        len_sents = len(sents)
        mask = [0] * len_sents
        sents_tokens = [self._tokenizer.tokenize(sent) for sent in sents]

        scores = []
        for sent_idx in range(len_sents):
            score = self.get_score_pretokenized(
                sents_tokens[:sent_idx]+sents_tokens[sent_idx+1:], 
                [sents_tokens[sent_idx]], 
                is_uniq)
            scores.append(score)
        max_idx = np.argmax(scores)
        mask[max_idx] = 1

        for _ in range(m-1):
            left_idx = max_idx - 1
            right_idx = max_idx + 1

            left_score = -1
            right_score = -1

            if left_idx >= 0:
                left_score = self.get_sent_score(left_idx, mask, sents_tokens, is_uniq)
            if right_idx < len_sents:
                right_score = self.get_sent_score(left_idx, mask, sents_tokens, is_uniq)
            
            max_idx = left_idx if left_score > right_score else right_idx
            mask[left_idx] = 1
        return mask


    def get_sent_score(self, idx, mask, sents_tokens, is_uniq=False):
        temp_mask = copy.deepcopy(mask)
        temp_mask[idx] = 1

        target = []
        prediction = []
        # keep order of sentences
        for t_idx, tm in enumerate(temp_mask):
            if tm == 0:
                target.append(sents_tokens[t_idx])
            else:
                prediction.append(sents_tokens[t_idx])
        return self.get_score_pretokenized(target, prediction, is_uniq)

    def get_ps_mask(self, sents, gsr=0.25, masking_type="ind-orig"):
        len_sents = len(sents)
        if len_sents < 2:
            raise ValueError(f"Number of sentences is too small to get principle sentences! [num of sents: {len_sents}]")
        m = max(1, int(len_sents*gsr))
        
        if masking_type == "lead":
            mask = np.zeros(len_sents, dtype=int)
            mask[:m] = 1
            return mask.tolist()
        elif masking_type == "random":
            mask = np.zeros(len_sents, dtype=int)
            index = np.random.choice(mask.shape[0], m, replace=False)
            mask[index] = 1
            return mask.tolist()
        elif masking_type == "ind-orig":
            return self.get_ps_ind(sents, m, is_uniq=False)
        elif masking_type == "ind-uniq":
            return self.get_ps_ind(sents, m, is_uniq=True)
        elif masking_type == "seq-orig":
            return self.get_ps_seq(sents, m, is_uniq=False)
        elif masking_type == "seq-uniq":
            return self.get_ps_seq(sents, m, is_uniq=True)
        else:
            raise ValueError(f"invalid masking_type: {masking_type}. masking_type must be chosen from lead, random, ind-orig, ind-uniq, seq-orig, seq-uniq.")

    def get_ps_mask_all(self, sents, gsr=0.25):
        return {
            k: self.get_ps_mask(sents, gsr, k) for k in [
                "lead", 
                "random", 
                "ind-orig", 
                "ind-uniq", 
                "seq-orig", 
                "seq-uniq"]
        }


def preproc_file(input_lines, output_path, args):
    tokenizer = args.tokenizer
    if tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    psc = PrincipleSentenceCreator(args.rouge_types)

    # fs = gcsfs.GCSFileSystem(project=args.project)
    # fo = fs.open(output_path, "w")

    fo = open(output_path, "w")
    for line in tqdm(input_lines):
        item = json.loads(line)
        sents = item["text"]
        if len(sents) < 2:
            continue

        ps_mask_all = psc.get_ps_mask_all(sents, gsr=args.gsr)
        for k, v in ps_mask_all.items():
            item[k] = v
        fo.write(json.dumps(item)+"\n")


def processing_wraper(inputs):
    input_lines, out_fname, args = inputs
    preproc_file(input_lines, out_fname, args)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def main():
    args = parse_args()

    def filter_fn(fname, start, end):
        basename = os.path.basename(fname)
        findex = int(basename.split(".")[1].split("-")[0])
        if findex >= start and findex <end:
            return True
        return False
    
    fs = gcsfs.GCSFileSystem(project=args.project)

    logger.info("get input files...")
    input_files = fs.glob(os.path.join(args.bucket, args.input_pattern))
    if args.start is not None and args.end is not None:
        filter_fn_p = functools.partial(filter_fn, start=args.start, end=args.end)
        input_files = list(sorted(filter(filter_fn_p, input_files)))
    
    # args.output_dir = os.path.join(args.bucket, args.output_dir)
    logger.info("create directory: {}".format(args.output_dir))
    # fs.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    num_procs = min(args.num_processes, cpu_count())
    progress_bar = tqdm(range(len(input_files)*num_procs), desc="overall")

    f_pairs = [
        (
            inp, 
            os.path.join(args.output_dir, os.path.basename(inp))
        ) for inp in input_files
    ]
    

    with Pool(num_procs) as p:
        
        logger.info("formatting inputs...")
        proc_inputs = []
        for inp, outf in f_pairs:
            fp = fs.open(inp, "r")
            inp_lines = fp.readlines()
            # inp_lines = open(inp, "r").readlines()
            logger.info("{} lines are added to the job queue".format(len(inp_lines)))
            inp_lines_sp = list(split(inp_lines, num_procs))

            tmp_proc_inputs = [
                (
                    inp,
                    "{}-{:03d}".format(outf, idx),
                    args) for idx, inp in enumerate(inp_lines_sp)
            ]
            proc_inputs.extend(tmp_proc_inputs)

        for _ in p.imap_unordered(processing_wraper, proc_inputs):
            progress_bar.update(1)


if __name__=="__main__":
    logging.basicConfig(level = logging.INFO)
    main()

