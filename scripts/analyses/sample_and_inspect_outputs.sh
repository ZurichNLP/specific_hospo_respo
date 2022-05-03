#!/usr/bin/env bash
# -*- coding: utf-8 -*-


raw_data=../data/hotel/500k
models=../models

line_number=$1

if [[ -z $line_number ]]; then
    max_lines=$(< "$raw_data/test.rrgen_id" wc -l) #$(( wc -l "$raw_data/test.rrgen_id" ))
    # echo "line count: $max_lines"
    line_number=$(( $RANDOM % $max_lines + 1 ))
    # echo $line_number
fi

rating=$( sed -n "${line_number}p" $raw_data/test.rating )
review=$( sed -n "${line_number}p" $raw_data/test.review )
response=$( sed -n "${line_number}p" $raw_data/test.response )

baseline=$( sed -n "${line_number}p" $models/baseline/inference/bs5.txt )
lex_freq=$( sed -n "${line_number}p" $models/filt_freq_distro/inference/bs5.txt )
sent_avg=$( sed -n "${line_number}p" $models/filt_gen_sent/inference/bs5.txt )
lm_ppl=$( sed -n "${line_number}p" $models/filt_tgt_ppl/inference/bs5.txt )


# echo -ne "$line_number;response;"
# sed -n "${line_number}p" $raw_data/test.response

# # differences model outputs
# for model_dir in baseline filt_freq_distro filt_gen_sent filt_tgt_ppl
# do
#     translations="$models/$model_dir/inference/bs5.txt"
#     echo -ne "$line_number;$translations;"
#     sed -n "${line_number}p" $translations
#     # echo ""
# done

# ## differences between labelled outputs
# for n in 0 1 2 3 4 5
# do
#     echo -ne "$line_number;ppl_${n};"
#     sed -n "${line_number}p" $models/label_tgt_ppl/inference/ppl${n}_bs5.txt
# done

# echo -ne "$line_number;filtered;"
# sed -n "${line_number}p" $models/filt_tgt_ppl/inference/bs5.txt

# echo "$line_number $rating"

echo "\begin{table*}[t!]"
echo "\begin{tabularx}{\textwidth}{| l | X |}"
echo "\hline"
echo "\multicolumn{2}{ | c | }{Review rating: $rating stars} \\\\ \hline"
echo "\textbf{Review} & $review \\\\ \hline"
echo "\textbf{Ground truth} & $response \\\\ \hline"
echo "\textbf{Baseline} & $baseline \\\\ \hline"
echo "\textbf{Lex. freq.} & $lex_freq \\\\ \hline"
echo "\textbf{Sent. avg.} & $sent_avg \\\\ \hline"
echo "\textbf{LM PPL} & $lm_ppl \\\\ \hline"
echo "\end{tabularx}"
echo "\caption{}"
echo "\label{tab:}"
echo "\end{table*}"