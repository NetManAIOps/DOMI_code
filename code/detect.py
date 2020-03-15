# -*- coding: utf-8 -*-
import numpy as np
from pot import POT


def pot_eval(init_score, q, level):
    """
    Run POT method on given score.
    init_score : The data to get init threshold. the outlier score of train set.
    q (float): Detection level (risk)
    level (float): Probability associated with the initial threshold t
    return the threshold under POT estimation algorithm.
    """
    s = POT(q)  # SPOT object
    pot_th = s.initialize(init_score, level=level)  # initialization step
    return pot_th


def cal_scoreChanges(outlierScore_list, ave_twoMetricScore = None, twoMetricScore = None):
    """
    get the change score of each metric
    return the list of outlier score change.
    """
    TwoMetricScoreList = []
    for i in range(0, len(outlierScore_list)):
        TwoMetricScoreList.append(-1*(np.array(twoMetricScore[i]) - np.array(ave_twoMetricScore)))
    return TwoMetricScoreList


def cal_binaryResult(outlierScore_list, threshold, timeIndex, saveMetricInfo = False,
    labelFileNameLineCntList = None):
    """
    output result according the threshold
    return the binary result whether it's an outlier.
    """
    result_dict = dict()
    fileNameLineCntList = []
    for i in range(0, len(outlierScore_list)):
        if outlierScore_list[i] < threshold:
            result_dict[i] = outlierScore_list[i]
            if saveMetricInfo:
                fileNameLineCntList.append(labelFileNameLineCntList[i])
    resultArray = [timeIndex[index] for index, value in result_dict.items()]
    if saveMetricInfo:
        return resultArray, fileNameLineCntList
    else:
        return resultArray

