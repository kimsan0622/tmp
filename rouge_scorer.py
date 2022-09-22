# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import logging
import collections
import six

import numpy as np
import nltk
from rouge_score import scoring
from rouge_score import tokenizers
from rouge_score.rouge_scorer import (
    _score_lcs,
    _summary_level_lcs,
    _score_ngrams
)


logger = logging.getLogger(__name__)


class RougeScorer(scoring.BaseScorer):
    """Calculate rouges scores between two blobs of text.
    Sample usage:
      scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
      scores = scorer.score('The quick brown fox jumps over the lazy dog',
                            'The quick brown dog jumps on the log.')
    """

    def __init__(self, rouge_types, use_stemmer=False, split_summaries=False,
                 tokenizer=None):
        """Initializes a new RougeScorer.
        Valid rouge types that can be computed are:
          rougen (e.g. rouge1, rouge2): n-gram based scoring.
          rougeL: Longest common subsequence based scoring.
        Args:
          rouge_types: A list of rouge types to calculate.
          use_stemmer: Bool indicating whether Porter stemmer should be used to
            strip word suffixes to improve matching. This arg is used in the
            DefaultTokenizer, but other tokenizers might or might not choose to
            use this.
          split_summaries: whether to add newlines between sentences for rougeLsum
          tokenizer: Tokenizer object which has a tokenize() method.
        Returns:
          A dict mapping rouge types to Score tuples.
        """

        self.rouge_types = rouge_types
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = tokenizers.DefaultTokenizer(use_stemmer)
            logger.info("Using default tokenizer.")

        self._split_summaries = split_summaries

    def score_multi(self, targets, prediction, is_uniq=False):
        """Calculates rouge scores between targets and prediction.
        The target with the maximum f-measure is used for the final score for
        each score type..
        Args:
          targets: list of texts containing the targets
          prediction: Text containing the predicted text.
        Returns:
          A dict mapping each rouge type to a Score object.
        Raises:
          ValueError: If an invalid rouge type is encountered.
        """
        score_dicts = [self.score(t, prediction, is_uniq) for t in targets]
        max_score = {}
        for k in self.rouge_types:
            index = np.argmax([s[k].fmeasure for s in score_dicts])
            max_score[k] = score_dicts[index][k]

        return max_score

    def score(self, target, prediction, is_uniq=False):
        """Calculates rouge scores between the target and prediction.
        Args:
          target: Text containing the target (ground truth) text,
          or if a list
          prediction: Text containing the predicted text.
        Returns:
          A dict mapping each rouge type to a Score object.
        Raises:
          ValueError: If an invalid rouge type is encountered.
        """

        # Pre-compute target tokens and prediction tokens for use by different
        # types, except if only "rougeLsum" is requested.
        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            target_tokens = None
            prediction_tokens = None
        else:
            target_tokens = self._tokenizer.tokenize(target)
            prediction_tokens = self._tokenizer.tokenize(prediction)
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                scores = _score_lcs(target_tokens, prediction_tokens)
            elif rouge_type == "rougeLsum":
                # Note: Does not support multi-line text.
                def get_sents(text):
                    if self._split_summaries:
                        sents = nltk.sent_tokenize(text)
                    else:
                        # Assume sentences are separated by newline.
                        sents = six.ensure_str(text).split("\n")
                    sents = [x for x in sents if len(x)]
                    return sents

                target_tokens_list = [
                    self._tokenizer.tokenize(s) for s in get_sents(target)]
                prediction_tokens_list = [
                    self._tokenizer.tokenize(s) for s in get_sents(prediction)]

                scores = _summary_level_lcs(target_tokens_list,
                                            prediction_tokens_list)
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError(
                        "rougen requires positive n: %s" % rouge_type)
                target_ngrams = _create_ngrams(target_tokens, n, is_uniq)
                prediction_ngrams = _create_ngrams(
                    prediction_tokens, n, is_uniq)
                scores = _score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores

        return result

    def score_pretokenized(self, target, prediction, is_uniq=False):
        """Calculates rouge scores between the target and prediction.
        Args:
          target: Text containing the target (ground truth) text,
          or if a list
          prediction: Text containing the predicted text.
        Returns:
          A dict mapping each rouge type to a Score object.
        Raises:
          ValueError: If an invalid rouge type is encountered.
        """

        # Pre-compute target tokens and prediction tokens for use by different
        # types, except if only "rougeLsum" is requested.
        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            target_tokens = None
            prediction_tokens = None
        else:
            target_tokens = [item for sublist in target for item in sublist]
            prediction_tokens = [item for sublist in prediction for item in sublist]
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                scores = _score_lcs(target_tokens, prediction_tokens)
            elif rouge_type == "rougeLsum":
                target_tokens_list = target
                prediction_tokens_list = prediction

                scores = _summary_level_lcs(target_tokens_list,
                                            prediction_tokens_list)
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError(
                        "rougen requires positive n: %s" % rouge_type)
                target_ngrams = _create_ngrams(target_tokens, n, is_uniq)
                prediction_ngrams = _create_ngrams(
                    prediction_tokens, n, is_uniq)
                scores = _score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores

        return result


def _create_ngrams(tokens, n, is_uniq=False):
    """Creates ngrams from the given list of tokens.
    Args:
      tokens: A list of tokens from which ngrams are created.
      n: Number of tokens to use, e.g. 2 for bigrams.
    Returns:
      A dictionary mapping each bigram to the number of occurrences.
    """

    ngrams = collections.Counter()
    for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
        if is_uniq:
            ngrams[ngram] = 1
        else:
            ngrams[ngram] += 1
    return ngrams
