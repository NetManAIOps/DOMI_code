# -*- coding: utf-8 -*-
from util import read_file, save_file
import numpy as np
import os


def evaluate(MLResult, labelpath, resultArray, timeIndex, threshold):
    """
    evalute the results
    return F score of prediction and truth.
    """
    groundTruthArray = []
    TPArray = []
    num = 0
    for fileName in read_file(MLResult.system_path("mid_summary"), "TestFileNameList.txt"):
        with open(labelpath + fileName,"r") as f:
            fline = f.readlines()
            for line in fline:
                count = line.strip("\n")
                if int(count) == 1 and num in timeIndex:
                    groundTruthArray.append(num)
                num += 1

    TP = 0
    for i in resultArray:
        if i in groundTruthArray:
            TP += 1
            TPArray.append(i)

    FP = len(resultArray) - TP
    FN = len(groundTruthArray) - TP
    Precision = TP / (float(TP + FP)) if TP + FP != 0 else 1
    Recall = TP/(float(TP + FN)) if TP+FN != 0 else 1
    F = 0 if Recall + Precision == 0 else (2 * Recall * Precision)/(Recall + Precision)
    save_file(
        MLResult.system_path("result_summary"), "PRF.txt",
        [threshold, F, Precision, Recall, TP, FP, FN, "\n"], ",", "a"
    )
    return F


def interpretation_hit_ratio(truth_filepath, prediction_filepath):
    """
    compute top 100%/120% interpretation hit ratio given truth lists of univariate time series
    that contribute to outlier judgement and predicted lists of univariate time series.
    return top 100%/120% interpretation hit ratio
    """
    with open(truth_filepath, 'r') as f:
        gt = f.readlines()

    with open(prediction_filepath, 'r') as f:
        result = f.readlines()

    gtDict = {}
    for i in gt:
        iList = i.strip("\n").strip("\r").split(",")
        gtDict[iList[0]] = iList[1:]

    resultDict = {}
    for i in result:
        iList = i.strip("\n").strip("\r").replace(".txt", "").split(",")
        resultDict[iList[0]] = iList[1:]

    for rate in [1.0, 1.2]:
        accurate_list = []
        for k in gtDict.keys():
            t1 = resultDict[k]
            t2 = gtDict[k]
            t3 = list(set(t2).intersection(t1[0:int(len(t2) * rate)]))
            accurate_list.append(float(len(t3)) / float(len(t2)))
        print("top {}% interpretation hit ratio: ".format(rate * 100), sum(accurate_list) / len(accurate_list))

