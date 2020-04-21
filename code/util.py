# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import csv
import numpy as np
from functools import partial
np.seterr(divide='ignore', invalid='ignore')
import multiprocessing as mul

delEXTREVALUE = True


def read_file(pathName, fileName, Type="string", name=False):
    """
    read the content from txt file to a matrix
    return the matrix of the file
    """
    matrix = []
    with open(os.path.join(pathName, fileName), "r") as f:
        lines = f.readlines()
        for line in lines:
            if "," in line:
                vector = line.strip("\r").strip("\n").split(',')
                matrix.append([float(v) for v in vector[1:]] if name else [float(v) for v in vector])
            else:
                if Type == "string":
                    matrix.append(line.strip("\r").strip("\n"))
                if Type == "float":
                    matrix.append(float(line.strip("\r").strip("\n")))
    return matrix


def save_file(pathName, fileName, resultList, cat="\n", writeType = "w"):
    """
    save the 'resultList' in a 'fileName' File
    """
    with open(os.path.join(pathName, fileName), writeType) as f:
        if len(np.array(resultList).shape) == 1:
            f.write(cat.join(str(x) for x in resultList) + '\n')
        if len(np.array(resultList).shape) == 2:
            w = csv.writer(f, delimiter=',')
            w.writerows(resultList)


def data_norm_all(path, dirPath, timeLength, metricNumber, norm):
    """
    use the entire metric for normalize
    return all normalized matrix data and filepath.
    """
    df = pd.read_csv(os.path.join(dirPath, path)).astype(float)
    matrix = np.array(df.values.tolist())
    matrix = np.around(matrix, decimals=2)
    if norm:
        if delEXTREVALUE:
            Y = np.sort(matrix, axis=0)
            a, _ = Y.shape
            Z = Y[int(0.01*a):int(0.99*a), :]
            m_mean = np.mean(Z, axis=0, keepdims=True).astype(float)
            m_std = np.std(Z, axis=0, keepdims=True).astype(float)
        else:
            m_mean = np.mean(matrix, axis=0).astype(float)
            m_std = np.std(matrix, axis=0).astype(float)

        norm_matrix = (matrix - m_mean) / m_std
        norm_matrix = np.where(np.isnan(norm_matrix), 0, norm_matrix)
        norm_matrix = np.around(norm_matrix, decimals=2)
        norm_matrix = norm_matrix.reshape(-1, timeLength*metricNumber)
        return norm_matrix.tolist(), path, norm_matrix.shape[0]
    else:
        matrix = np.around(matrix, decimals=2)
        matrix = np.array(matrix).reshape(-1, timeLength*metricNumber)
        return matrix.tolist(), path, matrix.shape[0]


def get_data_eachday(path, dirPath, timeLength, metricNumber, norm):
    """
    use the each day metric for normalize
    return normalized matrix data for each day and filepath.
    """
    df = pd.read_csv(os.path.join(dirPath, path)).astype(float)
    matrix = np.array(df.values.tolist())
    matrix = np.around(matrix, decimals=2)

    if norm:
        matrix = matrix.reshape(-1, timeLength, metricNumber)
        if delEXTREVALUE:
            Y = np.sort(matrix, axis=1)
            a,b,c = Y.shape
            Z = Y[:, int(0.01*b):int(0.99*b), :]
            m_mean = np.mean(Z, axis=1, keepdims=True).astype(float)
            m_std = np.std(Z, axis=1, keepdims=True).astype(float)
        else:
            m_mean = np.mean(matrix, axis=1, keepdims=True).astype(float)
            m_std = np.std(matrix, axis=1, keepdims=True).astype(float)

        norm_matrix = (matrix - m_mean) / m_std
        norm_matrix = np.where(np.isnan(norm_matrix), 0, norm_matrix)
        norm_matrix = np.where(norm_matrix > 2.5, 2.5, norm_matrix)
        norm_matrix = np.where(norm_matrix < -2.5, -2.5, norm_matrix)
        norm_matrix = (norm_matrix + 2.5) / 5.0
        norm_matrix = np.around(norm_matrix, decimals=2)
        norm_matrix = norm_matrix.reshape(-1, timeLength*metricNumber)
        return norm_matrix.tolist(), path, norm_matrix.shape[0]
    else:
        matrix = np.around(matrix, decimals=2)
        matrix = np.array(matrix).reshape(-1, timeLength*metricNumber)
        return matrix.tolist(), path, matrix.shape[0]


def load_matrix_allData(dataReadformat, dirPath1, dirPath2,
    timeLength, metricNumber, fileInfo1, fileInfo2, MLResult, norm):
    """
    read and normalize the data by Parallel using pool
    return the two matrix data and corresponding time index.
    """
    st = time.time()

    matrix1, matrix2 = [], []
    fileDirList1, fileDirList2 = [], []

    WORKERS = mul.cpu_count()
    pool = mul.Pool(processes=WORKERS, maxtasksperchild=WORKERS)

    paras1 = [path for path in os.listdir(dirPath1) if ".txt" in path]
    paras2 = [path for path in os.listdir(dirPath2) if ".txt" in path]

    if dataReadformat == "all":
        get_data_partial1 = partial(
            data_norm_all, dirPath=dirPath1, timeLength=timeLength, metricNumber=metricNumber, norm=norm
        )
        get_data_partial2 = partial(
            data_norm_all, dirPath=dirPath2, timeLength=timeLength, metricNumber=metricNumber, norm=norm
        )
        result1 = pool.map_async(get_data_partial1, paras1)
        result2 = pool.map_async(get_data_partial2, paras2)
    else:
        get_data_partial1 = partial(
            get_data_eachday, dirPath=dirPath1, timeLength=timeLength, metricNumber=metricNumber, norm=norm
        )
        get_data_partial2 = partial(
            get_data_eachday, dirPath=dirPath2, timeLength=timeLength, metricNumber=metricNumber, norm=norm
        )
        result1 = pool.map_async(get_data_partial1, paras1)
        result2 = pool.map_async(get_data_partial2, paras2)

    pool.close()
    pool.join()

    for i in result1.get():
        matrix1 += i[0]
        for j in range(1, i[2]+1):
            fileDirList1.append(i[1]+'+'+str(j))
    for i in result2.get():
        matrix2 += i[0]
        for j in range(1, i[2] + 1):
            fileDirList2.append(i[1]+'+'+str(j))
    time_indexs1 = [i for i in range(0, len(matrix1))]
    time_indexs2 = [i for i in range(0, len(matrix2))]
    save_file(MLResult.system_path("mid_summary"), fileInfo1, fileDirList1)
    save_file(MLResult.system_path("mid_summary"), fileInfo2, fileDirList2)
    print("-----Get data>>>>>:Time:%s" % (time.time()-st))

    return np.array(matrix1), np.array(time_indexs1), np.array(matrix2), np.array(time_indexs2)


def cat_List(a, b):
    """
    cat the a: fileNameLineCnt list and b: resultList
    return the combined list.
    """
    c = []
    i = 0
    while i < len(a):
        if isinstance(b[i], list):
            c.append([a[i]] + b[i])
        elif isinstance(b[i], np.ndarray):
            c.append([a[i]] + b[i].tolist())
        else:
            c.append([a[i]] + [b[i]])
        i += 1
    return c


def get_machineID(MLResult, labelpath=None):
    """
    get the list: fileName + lineNum
    return the result list.
    """
    if labelpath is None:
        return read_file(MLResult.system_path("mid_summary"), "TestFileNameList.txt")
    else:
        labelFileNameLineCntList = []
        for fileName in read_file(MLResult.system_path("mid_summary"), "TestFileNameList.txt"):
            with open(labelpath + fileName, "r") as f:
                fline = f.readlines()
                lineCnt = 1
                while lineCnt <= len(fline):
                    labelFileNameLineCntList.append(fileName+"+"+str(lineCnt))
                    lineCnt += 1
        return labelFileNameLineCntList

