#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict
import numpy as np
import fasttext
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# silence stupid warning message (https://github.com/facebookresearch/fastText/issues/1067)
fasttext.FastText.eprint = lambda x: None

domain_classifier = '../models/classifiers/fasttext/domain_classifier/domain.model.bin'
source_classifier = '../models/classifiers/fasttext/source_classifier/source.model.bin'
# rating_classifier = '../models/classifiers/fasttext/rating_classifier/rating.model.bin'
rating_classifier = '../models/classifiers/rating/fasttext.model.bin'

# NOTE: depending on the model, attribute labels may be floats (LSTM+) or tags (TF)
domain_mapping = {
    '1': '__label__restaurant',
    '0': '__label__hotel',
    '<restaurant>': '__label__restaurant',
    '<hotel>': '__label__hotel'
    }

# NOTE: depending on the model, attribute labels may be floats (LSTM+) or tags (TF)
ratings_mapping = {
    '1.0': '__label__5',
    '0.75': '__label__4',
    '0.5': '__label__3',
    '0.25': '__label__2',
    '0.0': '__label__1',
    '<5>': '__label__5',
    '<4>': '__label__4',
    '<3>': '__label__3',
    '<2>': '__label__2',
    '<1>': '__label__1',
    '5': '__label__POS',
    '4': '__label__POS',
    '3': '__label__NTR',
    '2': '__label__NEG',
    '1': '__label__NEG',
    }

source_mappings = {
    're': '__label__re',
    'tripadvisor': '__label__tripadvisor',
    'platform': '__label__tripadvisor' # NOTE: source classifier is trained only on re and
    # tripadvsior source texts. platform source texts are
    # considered tripadvisor for the sake of this experiment.
    }


def classify_texts(texts: List[str], model_path: str, k: int = 1) -> List[float]:
    model = fasttext.load_model(model_path)
    preds, conf = model.predict(texts, k=k)
    return preds

def compute_accuracy(y_true, y_pred):
    """
    y_pred is a list of tuples containing classifiers top-k predictions
    NOTE: for review rating, we generalise from a discrete scale to a range ([0,1], [1,2], [2,3], etc.)
    """
#     print(y_pred)
    assert not isinstance(y_pred[0], tuple)
    correct = []
    hit = len(y_pred[0])
    for truth, predictions in zip(y_true, y_pred):
        if truth in predictions:
            correct.append(1)
        else:
            correct.append(0)

    # more detailed analysis
    # y_true = np.array(y_true) 
    # y_pred = np.array([y[0] for y in y_pred])        
    # eq = y_true == y_pred
    # print("CLSFR Accuracy: " + str(eq.sum() / len(y_true)))
    # cm = confusion_matrix(y_true, y_pred)
    # print("CLSFR confusion matrix:")
    # print(cm)
    # print(precision_recall_fscore_support(y_true, y_pred, average='macro'))
    # print(precision_recall_fscore_support(y_true, y_pred, average='micro'))

    return f'{sum(correct) / len(correct):.3f} @ hit {hit}'

# def read_reference_labels(file):
#     with open(file, 'r', encoding='utf8') as f:
#         for line in f:
#             yield line.strip()

def estimate_domain_accuracy(
    refs: List[str],
    hyps: List[str],
    ground_truth_labels: List[str],
    model_path: str = domain_classifier,
    verbose: bool = False,
    ) -> Dict:
    """
    classify reference texts with multilingual domain
    classifier trained with fasttext

    args:
        refs: reference texts that are classified for
        accuracy comparison
        hyps: system output hypotheses to be classifies
        ground_truth_labels: true classification labels
    """
    
    ref_preds = classify_texts(list(map(lambda x : x.lower(), refs)),
                               model_path=model_path,
                               k=1)
    if verbose:
        print('REFS:', ref_preds[:10], '...')
    
    hyp_preds = classify_texts(list(map(lambda x : x.lower(), hyps)),
                               model_path=model_path,
                               k=1)

    if verbose:
        print('HYPS:', hyp_preds[:10], '...')

    if not ground_truth_labels:
        raise RuntimeError('Cannot compute accuracy - missing ground truth labels')
    else:
        ground_truth_labels = [domain_mapping[i] for i in ground_truth_labels]
    
    if verbose:
        print('TRUTH:', ground_truth_labels[:10], '...')

    
    return {
        'accuracy_on_refs': compute_accuracy(ground_truth_labels, ref_preds),
        'accuracy_on_hyps': compute_accuracy(ground_truth_labels, hyp_preds)
    }


def estimate_rating_accuracy(
    refs: List[str],
    hyps: List[str],
    ground_truth_labels: List[str],
    model_path: str = rating_classifier,
    verbose: bool = False,
    ) -> Dict:
    """
    classify reference texts with review rating classifier
    """
    
    ref_preds = classify_texts(list(map(lambda x : x.lower(), refs)),
                               model_path=model_path,
                               k=2)
    if verbose:
        print('REFS:', ref_preds[:10], '...')
    
    hyp_preds = classify_texts(list(map(lambda x : x.lower(), hyps)),
                               model_path=model_path,
                               k=2)
    
    if verbose:
        print('HYPS:', hyp_preds[:10], '...')

    if not ground_truth_labels:
        raise RuntimeError('Cannot compute accuracy - missing ground truth labels!')
    else:
        ground_truth_labels = [ratings_mapping[i] for i in ground_truth_labels]

    if verbose:
        print('TRUTH:', ground_truth_labels[:10], '...')

    return {
            'accuracy_on_refs': compute_accuracy(ground_truth_labels, ref_preds),
            'accuracy_on_hyps': compute_accuracy(ground_truth_labels, hyp_preds)
        }


def estimate_source_accuracy(
    refs: List[str],
    hyps: List[str],
    ground_truth_labels: List[str],
    model_path: str = source_classifier,
    verbose: bool = False,
    ) -> Dict:
    """
    classify reference texts according to source, NOTE: all sources should be re:spondelligent in RE_TEST!
    """
    
    ref_preds = classify_texts(list(map(lambda x : x.lower(), refs)),
                               model_path=model_path,
                               k=1)

    if verbose:
        print('REFS:', ref_preds[:10], '...')
        
    hyp_preds = classify_texts(list(map(lambda x : x.lower(), hyps)),
                               model_path=model_path,
                               k=1)

    if verbose:
        print('HYPS:', hyp_preds[:10], '...')

    if not ground_truth_labels: # assume all as respondelligent!
        ground_truth_labels = ['__label__re'] * len(ref_preds)
    else:
        # import pdb;pdb.set_trace()
        ground_truth_labels = [source_mappings[i] for i in ground_truth_labels]


    if verbose:
         print('TRUTH:', ground_truth_labels[:10], '...')
            
    return {
        'accuracy_on_refs': compute_accuracy(ground_truth_labels, ref_preds),
        'accuracy_on_hyps': compute_accuracy(ground_truth_labels, hyp_preds)
        }


if __name__ == '__main__':
    # test
    model_path = '/srv/scratch2/kew/classifiers/fasttext/domain_classifier/domain.model.bin'

    ex1 = """<greeting> thank you for your positive \
        feedback . it 's great to read you enjoyed our pizza . \
        like in a true <name> , we use the original <name> flour \
        for pizza dough and fior di latte instead of normal \
        mozzarella . this ensures an authentic taste appreciated \
        by many of our guests . we 'd love to welcome you back \
        to <name> for lunch or dinner . perhaps , next time you \
        'd like to try <digit> of our homemade pasta dishes . \
        <salutation>"""

    ex2 = """<greeting> thank you for the <digit> - star review . \
    we are happy to read you enjoyed your meal at santa \
    lucia niederdorf in the heart of <loc> 's old town . we \
    are pleased you enjoyed your meal at santa lucia \
    niederdorf in the heart of <loc> 's old town . we look \
    forward to welcoming you back to our restaurant in the \
    heart of <loc> . <salutation>"""

    ex3 = """<greeting> thank you for the <digit> - star review . \
    we are glad you enjoyed your stay at our hotel . we are \
    happy to hear that you enjoyed your stay at our hotel . \
    we are open every day and serve warm meals throughout \
    the day . we look forward to seeing you again when you \
    are next in <loc> . <salutation>"""

    ex4 = "thank you . thank you very much . thanks so much !"

    ex5 = "thank you . thank you very much ."

    ex6 = "thank you for taking the time to write a review . we hope to see you again soon ."

    preds = classify_texts([ex1, ex2, ex3, ex4, ex5, ex6], model_path, k=2)
    print(preds)