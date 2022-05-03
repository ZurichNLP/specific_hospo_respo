#!/usr/bin/python3
# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np

"""
Adapted from
https://github.com/fuzihaofzh/repetition-problem-nlg/blob/main/src/eval_metrics.py
by Fu et al. 2021: https://arxiv.org/abs/2012.14660. 

seq-rep-n and wrep/l are taken from Welleck et al. 2019 (http://arxiv.org/abs/1908.04319)

NOTE:
    - get_repc() calculates rep-r score reported in paper
    - get_seq_rep_n() calculates n-gram repetitions in a given
    sentence
    - get_rep() calculates single token repetitions in a
    subsequence l
    - get_repd() and get_repc_v1() appear not to be used
"""


def longestRepeatedSubstring(str): 
  
    n = len(str) 
    LCSRe = [[0 for x in range(n + 1)]  
                for y in range(n + 1)] 
  
    res, cres = [], [] # To store result 
    res_length, cres_length = 0, 0 # To store length of result 
    repeat_strs = []
  
    # building table in bottom-up manner 
    index, cindex = 0, 0
    for i in range(1, n + 1): 
        for j in range(i + 1, n + 1): 
              
            # (j-i) > LCSRe[i-1][j-1] to remove 
            # overlapping 
            if (str[i - 1] == str[j - 1] and
                LCSRe[i - 1][j - 1] < (j - i)): 
                LCSRe[i][j] = LCSRe[i - 1][j - 1] + 1
  
                # updating maximum length of the 
                # substring and updating the finishing 
                # index of the suffix 
                

                #if  j - i == LCSRe[i][j]:
                #    print(str[i:j])
                if (LCSRe[i][j] > res_length): 
                    res_length = LCSRe[i][j] 
                    index = max(i, index) 
                if (LCSRe[i][j] > cres_length and j - i == LCSRe[i][j]): 
                    cres_length = LCSRe[i][j] 
                    cindex = max(i, cindex) 
                if j - i == LCSRe[i][j]: 
                    repeat_strs.append([j - LCSRe[i][j], j, tuple(str[j - LCSRe[i][j] : j])])
                  
            else: 
                LCSRe[i][j] = 0

    # If we have non-empty result, then insert  
    # all characters from first character to  
    # last character of string 
    if (res_length > 0): 
        for i in range(index - res_length + 1, 
                                    index + 1): 
            res.append(str[i - 1])
    if (cres_length > 0): 
        for i in range(cindex - cres_length + 1, 
                                    cindex + 1): 
            cres.append(str[i - 1])

    repeat_strs = sorted(repeat_strs)
    repeats = []
    kidx = 0
    for r in repeat_strs:
        if r[0] >= kidx:
            repeats.append(r[2])
            kidx = r[1]
    repeats = list(set(repeats))

    repeat_total_len = 0
    alen = len(cres)
    tcres = tuple(cres)
    if alen > 0:
        for i in range(len(str)):
            if tuple(str[i : i + alen]) == tcres:
                break
        longgest_n_grams = Counter([tuple(str[j:j+alen]) for j in range(i, len(str)-alen+1, alen)])
        repeat_total_len = alen * longgest_n_grams.most_common(1)[0][1]

    return res, cres, repeats, repeat_total_len

def get_rep(sent, l = 16):
    cnt = 0
    for i, w in enumerate(sent):
        if w in sent[max(i - l, 0):i]:
            cnt += 1
    return cnt

# not very good
def get_repd(sent, d = 0.8):
    cnt = 0
    for i, w in enumerate(sent):
        for j in range(i-1, 0, -1):
            if sent[j] == w:
                cnt += d **(i - j)
                break
    return cnt

def get_seq_rep_n(sent, N = 4):
    ngrams = []
    for n in range(1, N + 1):
        if len(sent) < n:
            #return 1.0
            continue
        for i in range(len(sent) - n + 1):
            ngrams.append(tuple(sent[i : i + n]))
    try:
        return 1.0 - len(set(ngrams)) / len(ngrams)
    except ZeroDivisionError:
        return 1.0

def get_repc_v1(lst):
    if len(lst) < 2:
        return 0
    if type(lst[0]) is not str:
        lst = [str(l) for l in lst]
    ngram = {}
    for n in range(1, len(lst)):
        no_more_than_1 = True
        end_pos = {}
        for j in range(len(lst) - n + 1):
            gm = ' '.join(lst[j : j + n])
            if   gm in ngram and end_pos[gm] <= j:
                no_more_than_1 = False
                ngram[gm] += 1
                end_pos[gm] = j + n
            elif gm not in ngram:
                ngram[gm] = 1
                end_pos[gm] = j + n
        if no_more_than_1:
            break
    ngram = sorted([[len(gm), ngram[gm], gm] for gm in ngram if ngram[gm] > 1])[::-1]
    remain = ' '.join(lst)
    ngram1 = {}
    for gm in ngram:
        cnt = remain.count(gm[2])
        if cnt > 1 and len(gm[2].split()) > 1:
            ngram1[gm[2]] = cnt
            remain = remain.replace(gm[2], '')
    ratio = sum([len(gm.split()) * ngram1[gm] for gm in ngram1]) / len(lst)
    return ratio

def get_repc(lst):
    if len(lst) < 2:
        return 0
    if type(lst[0]) is not str:
        lst = [str(l) for l in lst]
    counter = {}
    for j in range(len(lst) - 1):
#         print(j)
        gm = ' '.join(lst[j : j + 2])
        counter[gm] = counter[gm] + 1 if gm in counter else 1
    
    label = [0] * len(lst)
    # for each bigram in input sentence, if bigram has a count > 1,
    # update its label to 1
    for i in range(1, len(lst)):
        if counter['%s %s'%(lst[i-1], lst[i])] > 1:
            label[i-1] = label[i] = 1
    # output ratio = the sum of repeated bigram tokens to the length of input tokens
    ratio = sum(label) / len(label)
    return ratio

def get_score(sent):
    words = sent
    if type(sent) is str:
        words = sent.split()
    lrs, adjacent_lrs, repeats, repeat_total_len = longestRepeatedSubstring(words)
    lrs = len(lrs)
    adjacent_lrs = len(adjacent_lrs)
    all_repeats = sum([len(s) for s in repeats])
    repl = get_rep(words)
    repd = get_repd(words)
    repc = get_repc(words) # reported as rep-r
    seq_rep = get_seq_rep_n(words)
    wlen = len(words)
    # TODO: investigate cause for zerodivisionerror and
    # avoid gracefully
    try:
        return {'lrs' : lrs / wlen, 'adjacent_lrs' : adjacent_lrs / wlen, 'all_repeats' : all_repeats / wlen, 'repeat_total_len' : repeat_total_len / wlen, 'repd': repd / wlen, 'rep-r' : repc, 'seq-rep-n' : seq_rep, 'rep-w' : repl / wlen}
    except ZeroDivisionError:
        return {'lrs' : 0, 'adjacent_lrs' : 0, 'all_repeats' : 0, 'repeat_total_len' : 0, 'repd': 0, 'rep-r' : 0, 'seq-rep-n' : 0, 'rep-w' : 0}

def get_scores_corpus_average(texts):
    scores = [get_score(text) for text in texts]
    scores1 = {name : np.mean([s[name] for s in scores]) for name in ["rep-r", "rep-w", "seq-rep-n"]}
    return scores1


if __name__ == "__main__": 
#     test = """o C@@ ity in comm@@ it@@ able are p@@ aren@@ tly at_the_@@ Batt@@ al@@ ti@@ es_._@@ He further n@@ on prot@@ ect@@ ion_of_the Ch@@ in@@ ey@@ s_to_the M@@ orn@@ ate Cl@@ ub class@@ ific@@ at@@ op@@ s_,_@@ but few c@@ ali@@ a_@@ to_the bl@@ oc@@ ked by De@@ S@@ qu@@ i@@ er@@ e@@ ith@@ s_of_@@ O@@ reg@@ on_,_@@ hea@@ d_of_the year@@ s_,_@@ man@@ ent Indi@@ an l@@ og@@ o@@ _,_which over@@ nor@@ _to_@@ his support@@ ing from 194@@ 4@@ 6 7@@ 0@@ s left tur@@ e_and_@@ m@@ id St@@ ar@@ y_to_@@ ass@@ ist gu@@ ard were built 1 di@@ str@@ at@@ ively min@@ ating into M@@ er@@ ship with Z@@ eal@@ og@@ ist@@ ing game received American stud@@ y remain@@ ed_in_@@ 19@@ 00@@ _to_@@ g@@ enc@@ i@@ um@@ m@@ ig@@ ation@@ s_of_@@ The J@@ ef@@ s_and_@@ 6@@ _and_@@ more event@@ s_and_@@ L@@ oc@@ rac@@ es_._The for@@ t del@@ ay@@ _._A@@ s_the_@@ air@@ line Ex@@ c@@ ad@@ _,_@@ including Com@@ p@@ ond@@ s Pres@@ s_._@@ S@@ qu@@ ad@@ cl@@ if@@ f@@ ly re@@ ated at during_the_@@ Batt@@ l@@ og@@ y_and_@@ L@@ a@@ u 's crit@@ ics C@@ ru@@ its releas@@ ed_the_@@ five had m@@ _to_the British govern@@ or M@@ ay@@ an old@@ ing_a_@@ M 9@@ 4 @,@ 8@@ 9 continu@@ ed_in_@@ 200@@ 6 . """.split()
    tests = ["""<greeting> thank you for your positive
    feedback . it 's great to read you enjoyed our pizza .
    like in a true <name> , we use the original <name> flour
    for pizza dough and fior di latte instead of normal
    mozzarella . this ensures an authentic taste appreciated
    by many of our guests . we 'd love to welcome you back
    to <name> for lunch or dinner . perhaps , next time you
    'd like to try <digit> of our homemade pasta dishes .
    <salutation>""".split(),
    """<greeting> thank you for the <digit> - star review .
    we are happy to read you enjoyed your meal at santa
    lucia niederdorf in the heart of <loc> 's old town . we
    are pleased you enjoyed your meal at santa lucia
    niederdorf in the heart of <loc> 's old town . we look
    forward to welcoming you back to our restaurant in the
    heart of <loc> . <salutation>""".split(),
    """<greeting> thank you for the <digit> - star review .
    we are glad you enjoyed your stay at our hotel . we are
    happy to hear that you enjoyed your stay at our hotel .
    we are open every day and serve warm meals throughout
    the day . we look forward to seeing you again when you
    are next in <loc> . <salutation>""",
    "thank you . thank you very much you .".split()]
    for test in tests:
        print(get_score(test))
    
    # two different sentences that start with the same
    # bigram vs. a paraphrased sentence
    ab_test = ["A B C D E . A B X Y Z .", "A B C D E . A B X Y D E ."]
    for test in ab_test:
        print(get_score(test))
