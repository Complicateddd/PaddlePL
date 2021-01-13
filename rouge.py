# -*- coding: utf-8 -*-
def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha=0.5):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0


def lcs(a, b):
    """
    Compute the length of the longest common subsequence between two sequences.
    Time complexity: O(len(a) * len(b))
    Space complexity: O(min(len(a), len(b)))
    """
    # This is an adaptation of the standard LCS dynamic programming algorithm
    # tweaked for lower memory consumption.
    # Sequence a is laid out along the rows, b along the columns.
    # Minimize number of columns to minimize required memory
    if len(a) < len(b):
        a, b = b, a
    # Sequence b now has the minimum length
    # Quit early if one sequence is empty
    if len(b) == 0:
        return 0
    # Use a single buffer to store the counts for the current row, and
    # overwrite it on each pass
    row = [0] * len(b)
    for ai in a:
        left = 0
        diag = 0
        for j, bj in enumerate(b):
            up = row[j]
            if ai == bj:
                value = diag + 1
            else:
                value = max(left, up)
            row[j] = value
            left = value
            diag = up
    # Return the last cell of the last row
    return left

def rouge_l(ground_truth, gen_str):
    """
    Compute the ROUGE-L score of a peer with respect to one or more models.
    """

    matches = lcs(gen_str, ground_truth)
    recall_total = len(gen_str)
    precision_total = len(ground_truth)
    return _safe_f1(matches, recall_total, precision_total)

import pandas as pd
pre = pd.read_csv(r'/media/ubuntu/data/OCRTijian/submit.csv')
answer = pd.read_csv(r'/media/ubuntu/data/OCRTijian/foreword/gpu/model/ocr_train_answer.csv')
df_all = pd.merge(answer, pre, on='id', how='left')
df = df_all.dropna(axis=0)
print(df)
score = 0
for i in range(len(df)):
    # print(df['id'][i],rouge_l(df['content_x'].values[i], df['content_y'].values[i]))
    print(i+1,df['content_x'].values[i])
    print(df['content_y'].values[i])
    print(rouge_l(df['content_x'].values[i], df['content_y'].values[i]))

    score += rouge_l(df['content_x'].values[i], df['content_y'].values[i])
print(score/len(df_all))