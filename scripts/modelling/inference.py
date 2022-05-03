#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is adapted from AllenAI's Longformer
TriviaQA task:
    https://github.com/allenai/longformer/blob/master/scripts/

Note: 
    Annette Rios (arios@cl.uzh.ch) initially adapted it for long-document simplication.
    Tannon Kew (kew@cl.uzh.ch) made minor changes for its
    application in the ReAdvisor project for response
    generation.
    
Date: 05/07/2021


known issues:
    /home/user/kew/anaconda3/envs/hospo_respo_bart/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  /pytorch/aten/src/ATen/native/BinaryOps.cpp:467.)
    return torch.floor_divide(self, other)



"""

import os
import argparse
import numpy as np
import json
import re
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import datasets

from train import get_eval_scores, CustomDataset

from memory_profiler import profile

import warnings
# warnings.filterwarnings("error")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_gpu_stats():
    
    no_gpus = torch.cuda.device_count()
    gpu_idx = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_idx)
    gpu_mem = torch.cuda.memory_stats(gpu_idx)

    stats = {
        "gpu_count": no_gpus,
        "gpu_idx": gpu_idx,
        "gpu_name": gpu_name,
        "gpu_mem": gpu_mem,
        }
    
    return stats

def score_ppl_for_batch_sequences(lprobs, target, epsilon, ignore_index=-100):
    """
    Compute entropy of a target sequence

    Adapted from label_smoothed_nll_loss function in
    `train.py`
    """

    # get target token probabilities 
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    target_probs = lprobs.gather(dim=-1, index=target)

    # replace any probs corresponding to padding token with 0
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target_probs.masked_fill_(pad_mask, 0.0)
    else:
        target_probs = target_probs.squeeze(-1)

    # compute perplexity for a sequence (same method as example in fairseq)
    # https://github.com/pytorch/fairseq/tree/master/examples/language_model
    # NOTE: pad_mask is originally True when element == 0,
    # so invert it here in order to normalise by true length
    # of example.
    batch_ppl = ((~pad_mask * target_probs).sum(dim=1) / (~pad_mask).sum(dim=1)).neg().exp()
    # compute entropy for each target sequence in the batch
    # target_probs has shape: [bsz, seq_len, 1]
    # batch_entropy = torch.distributions.Categorical(probs=target_probs.squeeze(-1)).entropy().unsqueeze(-1)
    # batch_entropy has shape: [bsz, 1]

    # normalise by length of target sequences
    # batch_entropy_norm = batch_entropy / torch.count_nonzero(target_probs, dim=1)

    return batch_ppl

class InferenceSimplifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
       
        self.src_lang = self.args.src_lang
        self.tgt_lang = self.args.tgt_lang
        
        self.config = BartConfig.from_pretrained(self.args.model_path)
        self.tokenizer = BartTokenizer.from_pretrained(self.args.model_path, use_fast=True)
        
        self.max_input_len = self.args.max_input_len if self.args.max_input_len is not None else self.config.max_encoder_position_embeddings
        self.max_output_len = self.args.max_output_len if self.args.max_output_len is not None else self.config.max_decoder_position_embeddings 
        self.test_dataloader_object = None
    
    # @profile
    def test_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        if not self.args.score_sequences: # do regular decoding

            input_ids, attention_mask, output_ids = batch
            batch_size = input_ids.shape[0]

            generated_outputs = self.model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                use_cache=True, 
                max_length=self.args.max_output_len, 
                num_beams=self.args.beam_size, 
                pad_token_id=self.tokenizer.pad_token_id, 
                decoder_start_token_id = self.tokenizer.pad_token_id,
                do_sample=self.args.do_sample,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                repetition_penalty=self.args.repetition_penalty,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True)

            # logging.info(torch.cuda.memory_summary())

            ref_strs = None
            if self.args.test_target and output_ids is not None:
                ref_strs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                            
            src_strs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
            gen_sequences = generated_outputs.sequences[:, 1:] # trim <bos> pad
            gen_strs = self.tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)
            
            # beam search outputs include pre-computed sequence scores
            # if sampling, sequence scores need to be computed manually
            gen_scores = generated_outputs.get('sequences_scores', None)
            if gen_scores is None:
                # https://discuss.huggingface.co/t/announcement-generationoutputs-scores-attentions-and-hidden-states-now-available-as-outputs-to-generate/3094/2
                # stack the logits generated at each step to
                # a tensor and transform logits to probs
                lprobs = torch.stack(generated_outputs.scores, dim=1).softmax(-1).log() # -> shape [num_gen, seq_len, vocab_size]
                # collect the probability of the generated token
                # add a dummy dim in the end to make gather work
                gen_lprobs = torch.gather(lprobs, 2, gen_sequences[:, :, None]).squeeze(-1)
                # sum up token level probabilities and
                # normalise by length, ignoring padded positions
                mask = (gen_lprobs != np.NINF).float()
                gen_scores = torch.nansum(gen_lprobs * mask, dim=-1) / mask.sum(dim=-1)

            batch_top1_hyps = []
            batch_gen_strs = np.array_split(gen_strs, batch_size)
            batch_gen_scores = np.array_split(gen_scores.cpu(), batch_size)

            for batch_idx in range(batch_size):
                
                output_dict = {
                    'src': src_strs[batch_idx],
                    'ref': ref_strs[batch_idx] if ref_strs is not None else None,
                    'hyps': [],
                    }
                
                # Ensure output hyps are sorted by their log-prob scores
                for i, idx in enumerate(batch_gen_scores[batch_idx].argsort(descending=True)):
                    if i == 0: # add the 1-best hypothesis to gen_strs for evaluation
                        batch_top1_hyps.append(batch_gen_strs[batch_idx][idx])
                    
                    output_dict['hyps'].append(
                        {'score': batch_gen_scores[batch_idx][idx].item(),
                        'hyp': batch_gen_strs[batch_idx][idx]
                        })
                
                json_line = json.dumps(output_dict, ensure_ascii=False)
            
                if self.args.output_to_json: # write source, ref and hyps as a dict - 1 set per line
                    with open(self.args.translation, 'a') as f:
                        f.write(json_line+"\n")
                else: # each hyp as a str - 1 hyp per line
                    with open(self.args.translation, 'a') as f:
                        for d in output_dict['hyps']:
                            f.write(d['hyp'] + "\n")
                
            if self.args.test_target is not None:                    
                return get_eval_scores(ref_strs, batch_top1_hyps)
            else:
                return {'decoded' : batch_top1_hyps}

        else:
            batch_ppl = self.score_sequences(batch).squeeze().tolist()
            input_ids, attention_mask, output_ids = batch
            batch_src_strs = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)
            batch_tgt_strs = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
            with open(self.args.translation, 'a') as f:
                for s, t, ppl in zip(batch_src_strs, batch_tgt_strs, batch_ppl):
                    f.write(f'{ppl:.5f}\t{s}\t{t}\n')
            return

    def score_sequences(self, batch):

        self.model.eval()

        input_ids, attention_mask, output_ids = batch

        decoder_input_ids = output_ids[:, :-1].clone() # without eos/last pad
        labels = output_ids[:, 1:].clone() # without bos
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)

        outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False
                )

        lm_logits = outputs[0]
        lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
        
        ppl_scores = score_ppl_for_batch_sequences(
            lprobs, labels, False, ignore_index=self.tokenizer.pad_token_id
        )

        return ppl_scores

    def test_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True
        
        if outputs:
            if self.args.test_target is not None:
                names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
                metrics = []
                for name in names:
                    scores = [x[name] for x in outputs]
                    metric = sum(scores)/len(scores)
                    metrics.append(metric)
                logs = dict(zip(*[names, metrics]))
                print("Evaluation on provided reference [{}] ".format(self.args.test_target))
                print(logs)
        else:
            pass

    def forward(self):
        pass
    
    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        reference = None
        if self.args.test_target is not None:
            reference = self.datasets[split_name + "_target"]
        
        if not self.args.reverse_sequences:
            dataset = CustomDataset(
                inputs=self.datasets[split_name + "_source"],
                labels=reference,
                name=split_name,
                tokenizer=self.tokenizer,
                max_input_len=self.max_input_len,
                max_output_len=self.max_output_len, 
                src_lang=self.src_lang, 
                tgt_lang=self.tgt_lang
                )
        else:
            # reverse sequences for scoring p(s|t)
            dataset = CustomDataset(
                inputs=reference,
                labels=self.datasets[split_name + "_source"],
                name=split_name,
                tokenizer=self.tokenizer,
                max_input_len=self.max_input_len,
                max_output_len=self.max_output_len, 
                src_lang=self.src_lang, 
                tgt_lang=self.tgt_lang
                )
      
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if self.trainer.use_ddp else None

        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=CustomDataset.collate_fn)

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model


    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--model_path", type=str, help="Path to the checkpoint directory or model name")
        parser.add_argument("--checkpoint_name", type=str, help="Checkpoint in model_path to use.")
        parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory.")
        
        #data
        parser.add_argument("--test_source", type=str, default=None, help="Path to the source test file.")
        parser.add_argument("--test_target", type=str, default=None, help="Path to the target test file (optional, if given, will output rouge and bleu).")
        parser.add_argument("--target_tags", type=str, default=None, help="If test_target is not given: provide path to file with list of target tags (one per sample in test_source).")
        parser.add_argument("--src_lang", type=str, default=None, help="Source language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tgt_lang", type=str, default=None, help="Target language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tags_included", action='store_true', help="Text files already contain special tokens (language tags and </s>. Source:  src_tag seq, Target:  tgt_tag seq. Note: actual source sequence is seq src_tag </s>, will be changed internally after possibly clipping sequence to given max_length.")
        parser.add_argument("--infer_target_tags", action="store_true", default=False, help="If test_target is not given and target language tags can be inferred from the source language tags provided with --tags_included (e.g. de_DE -> de_DE). This save having a dedicated text file in which the tags are explicitly specified.")
        parser.add_argument("--max_input_len", type=int, default=512, help="maximum num of wordpieces, if unspecified, will use number of encoder positions from model config.")
        parser.add_argument("--max_output_len", type=int, default=512, help="maximum num of wordpieces, if unspecified, will use number of decoder positions from model config.")

        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--gpus", type=int, default=1, help="Number of gpus. 0 for CPU")
        
        ## inference params
        parser.add_argument("--translation", type=str, default='decoded.out', help="Output file to write decoded sequence to.")
        parser.add_argument("--beam_size", type=int, default=1, help="Beam size for inference when testing/validating. Default: 4.")
        parser.add_argument("--test_percent_check", default=1.00, type=float, help='Percent of test data used')
        # parser.add_argument("--global_attention_indices", type=int, nargs='+', default=[-1], required=False, help="List of indices of positions with global attention for longformer attention. Supports negative indices (-1 == last non-padding token). Default: [-1] == last source token (==lang_id) .")

        parser.add_argument("--output_to_json", default=False, action="store_true", help='If true, decoding output is a verbose JSONL containing, src, tgt, and scored model output hyps')
        
        # decoding strategy params (passed to model.generate() (in generation_utils.py))
        parser.add_argument("--do_sample", default=False, action="store_true", help='Whether or not to use sampling ; use greedy decoding otherwise.')
        parser.add_argument("--temperature", default=1.0, type=float, help='The value used to module the next token probabilities.')
        parser.add_argument("--top_k", default=0, type=int, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
        parser.add_argument("--top_p", default=1.0, type=float, help='If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are kept for generation.')
        parser.add_argument("--repetition_penalty", default=1.0, type=float, help='The parameter for repetition penalty. 1.0 means no penalty.')
        parser.add_argument("--length_penalty", default=1.0, type=float, help='Exponential penalty to the length. 1.0 means no penalty.')
        parser.add_argument("--num_return_sequences", default=1, type=int, help='The number of independently computed returned sequences for each element in the batch, i.e. N-best')
        # parser.add_argument("--output_scores", default=False, action="store_true", help='Whether or not to return the prediction scores.')
        # parser.add_argument("--return_dict_in_generate", default=False, action="store_true", help='Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.')
        
        #logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")
        
        parser.add_argument("--score_sequences", action='store_true', help="If provided, loaded model is used to score target sequences. No decoding is performed.")
        parser.add_argument("--reverse_sequences", action='store_true', help="Allows scoring target-source pairs instead of source-target.")
        
        parser.add_argument("--wandb", type=str, default=None, help="WandB project name to use if logging fine-tuning with WandB.")

        return parser


def main(args):

    if Path(args.translation).is_file():
        logging.info("Output file `{}` already exists and will be overwritten...".format(args.translation))
        Path(args.translation).unlink()

    checkpoint_path=os.path.join(args.model_path, args.checkpoint_name)
    simplifier = InferenceSimplifier(args)
    
    cp = torch.load(checkpoint_path)
    simplifier.model = BartForConditionalGeneration.from_pretrained(args.model_path)
   
    simplifier.load_state_dict(cp["state_dict"])
     
    if args.print_params:
        for name, param in simplifier.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)
    
    if args.test_target is not None:
        simplifier.datasets = datasets.load_dataset('text', data_files={'test_source': args.test_source, 'test_target': args.test_target })
    else:
        if args.tags_included and args.infer_target_tags:
            # NOTE: tags_included expects input sequences to
            # be prefixed with a single language tag, e.g. de_DE.
            # language tag must be separable by whitespace!,
            # e.g. de_DE\s, en_XX\s, etc.
            data_dict = datasets.load_dataset('text', data_files={'test_source': args.test_source})
            # datasets library allows loading from an
            # in-memory dict, so construct one from the source
            # text tags that can be loaded
            target_tags_dict = {'text': [text.split()[0] for text in data_dict['test_source']['text']]} 
            data_dict['target_tags'] = datasets.Dataset.from_dict(target_tags_dict)
            simplifier.datasets = data_dict
        elif args.target_tags is not None:
            simplifier.datasets = datasets.load_dataset('text', data_files={'test_source': args.test_source, 'target_tags': args.target_tags })
        else:
            simplifier.datasets = datasets.load_dataset('text', data_files={'test_source': args.test_source })

    if args.wandb:
        logger = WandbLogger(project=args.wandb, job_type="evaluate")
    else:
        logger = TestTubeLogger(
            save_dir=".",
            name="decode.log",
            version=0  # always use version=0
        )

    # logging.info(torch.cuda.memory_summary())
    
    logging.info(get_gpu_stats())

    trainer = pl.Trainer(gpus=args.gpus, accelerator='ddp' if torch.cuda.is_available() else None,
                         replace_sampler_ddp=False,
                         limit_test_batches=args.test_percent_check,
                         logger=logger,
                         progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                         precision=32 if args.fp32 else 16, amp_level='O2'
                         )
    
    # if args.score_sequences:
    #     simplifier.score_sequences()
    logging.info(get_gpu_stats())
    # else:
    trainer.test(simplifier)

    print("Decoded outputs written to {}".format(args.translation))
        

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="simplification")
    parser = InferenceSimplifier.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)

