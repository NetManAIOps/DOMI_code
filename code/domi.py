# -*- coding: utf-8 -*-
import functools
import sys
import os
import time
import numpy as np
np.set_printoptions(precision=2)
from argparse import ArgumentParser
import tensorflow as tf
from pprint import pformat
from tensorflow.contrib.framework import arg_scope

import tfsnippet as spt
from tfsnippet.dataflows import DataFlow
from tfsnippet.scaffold import CheckpointSaver
from tfsnippet.utils import split_numpy_array, get_batch_size
from tfsnippet.examples.utils import MLResults, print_with_title, MultiGPU

from util import save_file, read_file, load_matrix_allData, get_machineID, cat_List
from evaluate import evaluate, interpretation_hit_ratio
from detect import pot_eval, cal_binaryResult, cal_scoreChanges
from model import q_net, p_net

from config import ExpConfig
config = ExpConfig()


def main():
    # parse the arguments
    arg_parser = ArgumentParser()
    spt.register_config_arguments(config, arg_parser, title='Model options')
    spt.register_config_arguments(spt.settings, arg_parser, prefix='tfsnippet', title='TFSnippet options')
    arg_parser.parse_args(sys.argv[1:])

    # print the config
    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    # open the result object and prepare for result directories
    model_file = config.result_dir + "/" + os.path.basename(__file__).split(".py")[0] + "_" + \
                 str(config.noExp) + ".model"
    dirName = os.path.basename(__file__).split(".py")[0] + "_" + str(config.noExp)
    results = MLResults(os.path.join(config.result_dir, dirName))
    results.save_config(config)  # save experiment settings
    results.make_dirs('train_summary', exist_ok=True)
    results.make_dirs('result_summary', exist_ok=True)
    results.make_dirs('mid_summary', exist_ok=True)

    # os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_number

    # input placeholders
    input_x = tf.placeholder(dtype=tf.float32, shape=(None,) + config.x_shape, name='input_x')
    learning_rate = spt.AnnealingVariable('learning_rate', config.initial_lr, config.lr_anneal_factor, min_value=1e-6)
    multi_gpu = MultiGPU(disable_prebuild=True)
    # multi_gpu = MultiGPU()

    # derive the training operation
    gradses = []
    grad_vars = []
    train_losses = []
    BATCH_SIZE = get_batch_size(input_x)

    for dev, pre_build, [dev_input_x] in multi_gpu.data_parallel(BATCH_SIZE, [input_x]):
        with tf.device(dev), multi_gpu.maybe_name_scope(dev):
            # derive the loss for initializing
            with tf.name_scope('initialization'), \
                    arg_scope([p_net, q_net], is_initializing=True), \
                    spt.utils.scoped_set_config(spt.settings, auto_histogram=False):
                init_q_net = q_net(dev_input_x, n_z=config.train_n_samples)
                init_chain = init_q_net.chain(p_net, latent_axis=0, observed={'x': dev_input_x})
                init_loss = tf.reduce_mean(init_chain.vi.training.vimco())

            # derive the loss and lower-bound for training
            with tf.name_scope('training'), \
                    arg_scope([p_net, q_net], is_training=True):
                train_q_net = q_net(dev_input_x, n_z=config.train_n_samples)
                train_chain = train_q_net.chain(p_net, latent_axis=0, observed={'x': dev_input_x})
                train_loss = (
                    tf.reduce_mean(train_chain.vi.training.vimco()) +
                    tf.losses.get_regularization_loss()
                )
                train_losses.append(train_loss)

            # derive the logits output for testing
            with tf.name_scope('testing'):
                test_q_net = q_net(dev_input_x, n_z=config.test_n_z)
                test_chain = test_q_net.chain(p_net, latent_axis=0, observed={'x': dev_input_x})
                # log_prob of X and each univariate time series of X
                log_prob = tf.reduce_mean(test_chain.model['x'].distribution.log_prob(dev_input_x), 0)
                log_prob_per_element = tf.reduce_sum(log_prob)
                log_prob_per_element_univariate_TS = tf.reduce_sum(log_prob, [0, 1, 3])
                log_prob_per_element_univariate_TS_All = tf.reduce_sum(log_prob, [1, 3])

            # derive the optimizer
            with tf.name_scope('optimizing'):
                params = tf.trainable_variables()
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads = optimizer.compute_gradients(train_loss, params)
                for grad, var in grads:
                    if grad is not None and var is not None:
                        if config.grad_clip_norm:
                            grad = tf.clip_by_norm(grad, config.grad_clip_norm)
                        if config.check_numerics:
                            grad = tf.check_numerics(grad, 'gradient for {} has numeric issue'.format(var.name))
                        grad_vars.append((grad, var))
                gradses.append(grad_vars)

    # merge multi-gpu outputs and operations
    [train_loss] = multi_gpu.average([train_losses], BATCH_SIZE)
    train_op = multi_gpu.apply_grads(
        grads=multi_gpu.average_grads(gradses),
        optimizer=optimizer,
        control_inputs=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    )

    # sort the contribution of each univariate_TS of input
    SORT_UNIVARIATE_TS_INPUT = tf.placeholder(dtype=tf.float32, shape=(None, None), name='SORT_UNIVARIATE_TS_INPUT')
    SORT_UNIVARIATE_TS = tf.nn.top_k(SORT_UNIVARIATE_TS_INPUT, k=config.metricNumber).indices + 1

    # load the training and testing data
    print("="*10+"Shape of Input data"+"="*10)
    x, time_indexs, x_test, time_indexs2 = load_matrix_allData(
        config.dataReadformat, config.datapathForTrain, config.datapathForTest, config.timeLength, config.metricNumber,
        "TrainFileNameList.txt", "TestFileNameList.txt", results, config.norm
    )

    x_test = x_test.reshape([-1, config.timeLength, config.metricNumber, 1])
    print("Test:", x_test.shape)
    if config.batchTest:
        test_flow = DataFlow.arrays([x_test], config.test_batch_size) # DataFlow is iterator
        del x_test
    x_train, x_val = split_numpy_array(x, portion=config.VALID_PORTION)
    x_train = x_train.reshape([-1, config.timeLength, config.metricNumber, 1])
    x_val = x_val.reshape([-1, config.timeLength, config.metricNumber, 1])
    train_flow = DataFlow.arrays([x_train], config.batch_size, shuffle=False, skip_incomplete=True)
    val_flow = DataFlow.arrays([x_val], config.test_batch_size)
    print("Note:", config.x_dim, ", x_dim = size of datapoint = timeLength * metricNumber")
    print("Input data shape:", x.shape, "Train data shape:", x_train.shape, "Validation data shape:", x_val.shape)
    del x_train, x_val, x

    # training part
    with spt.utils.create_session().as_default() as session:
        spt.utils.ensure_variables_initialized()
        saver = CheckpointSaver(tf.trainable_variables(), model_file)
        if os.path.exists(model_file):
            # load the parameters of trained model
            saver.restore_latest()
        else:
            # initialize the network
            while True:
                breakFlag = 0
                for [x] in train_flow:
                    INITLOSS = session.run(init_loss, feed_dict={input_x: x})
                    print('Network initialized, first-batch loss is {:.6g}.'.format(INITLOSS))
                    if np.isnan(INITLOSS) or np.isinf(INITLOSS) or INITLOSS > 10 ** 5:
                        pass
                    else:
                        breakFlag = 1
                        break
                if breakFlag:
                    break

            # train the network
            with train_flow.threaded(10) as train_flow:
                with spt.TrainLoop(params,
                                   var_groups=['q_net', 'p_net'],
                                   max_epoch=config.max_epoch,
                                   max_step=config.max_step,
                                   summary_dir=(results.system_path('train_summary') if config.write_summary else None),
                                   summary_graph=tf.get_default_graph(),
                                   early_stopping=True) as loop:
                    trainer = spt.Trainer(
                        loop, train_op, [input_x], train_flow,
                        metrics={'loss': train_loss},
                        summaries=tf.summary.merge_all(spt.GraphKeys.AUTO_HISTOGRAM)
                    )
                    # anneal the learning rate
                    trainer.anneal_after(
                        learning_rate,
                        epochs=config.lr_anneal_epoch_freq,
                        steps=config.lr_anneal_step_freq
                    )
                    validator = spt.Validator(
                        loop, train_loss, [input_x], val_flow,
                    )
                    trainer.evaluate_after_epochs(validator, freq=10)
                    trainer.log_after_epochs(freq=1)
                    trainer.run()
                saver.save()

            # save the training infomation
            firWrite = True
            num = 0
            time0 = time.time()
            for [x_train] in train_flow:
                if config.savetrainDS:
                    # log prob of each metric of each instance
                    log_prob_per_element_univariate_TS_list_item_Train = (session.run(
                        log_prob_per_element_univariate_TS_All,
                        feed_dict={input_x: x_train}
                    ))
                    log_prob_per_element_univariate_TS_list_Train = log_prob_per_element_univariate_TS_list_item_Train
                    log_prob_per_element_list_Train = np.sum(
                        np.array(log_prob_per_element_univariate_TS_list_item_Train), axis=1
                    ).tolist()
                    if firWrite:
                        save_file(
                            results.system_path("train_summary"), "OutlierScores_metric.txt",
                            log_prob_per_element_univariate_TS_list_Train
                        )
                        save_file(
                            results.system_path("train_summary"), "OutlierScores.txt", log_prob_per_element_list_Train)
                    else:
                        save_file(
                            results.system_path("train_summary"), "OutlierScores_metric.txt",
                            log_prob_per_element_univariate_TS_list_Train, "\n", "a"
                        )
                        save_file(
                            results.system_path("train_summary"), "OutlierScores.txt",
                            log_prob_per_element_list_Train, "\n", "a"
                        )

                firWrite = False
                num += 1
                if num % 1000 == 0:
                    print(
                        "-----Train %s >>>>>:Sum time of batch instances:%s" %
                        (num, float(time.time()-time0)/float(num))
                    )
            del train_flow, val_flow

        # online test
        time2 = time.time()
        log_prob_per_element_list, log_prob_per_element_univariate_TS_list = [], []
        if config.batchTest:
            num = 0
            for [x_test] in test_flow:
                if config.savetestDS:
                    # log prob of each metric of each instance
                    log_prob_per_element_univariate_TS_list_item = (session.run(
                        log_prob_per_element_univariate_TS_All,
                        feed_dict={input_x: x_test}
                    ))
                    log_prob_per_element_univariate_TS_list += log_prob_per_element_univariate_TS_list_item.tolist()
                    log_prob_per_element_list += np.sum(np.array(
                        log_prob_per_element_univariate_TS_list_item), axis=1
                    ).tolist()

                num += 1
                if num % 200 == 0:
                    print(
                        "-----Test %s >>>>>:Sum time of batch instances:%s" %
                        (num, float(time.time()-time2)/float(num))
                    )
        else:
            num = 1
            for batch_x in x_test:
                if config.savetestTS:
                    log_prob_per_element_list_item = (session.run(log_prob_per_element, feed_dict={input_x: [batch_x]}))
                    log_prob_per_element_list.append(log_prob_per_element_list_item)

                if config.savetestDS:
                    log_prob_per_element_univariate_TS_list_item = (session.run(
                        log_prob_per_element_univariate_TS,
                        feed_dict={input_x: [batch_x]}
                    ))
                    log_prob_per_element_univariate_TS_list.append(log_prob_per_element_univariate_TS_list_item)
                    log_prob_per_element_list.append(sum(log_prob_per_element_univariate_TS_list_item))

                if num % 200 == 0:
                    print(
                        "-----Test>>>>>:%d, average time of each instance:%s" %
                        (num, float(time.time()-time2)/float(num))
                    )
                num += 1

        # get the lable file name and its line cnt number
        allLabelFileNameLineCntList = get_machineID(results, config.labelpath)

        print("No of OutlierScores for all dataPoint:(%s):" % len(log_prob_per_element_list))
        if config.savetestDS:
            save_file(
                results.system_path("result_summary"), "OutlierScores_metric.txt",
                cat_List(allLabelFileNameLineCntList, log_prob_per_element_univariate_TS_list)
            )
        save_file(
            results.system_path("result_summary"), "OutlierScores.txt",
            cat_List(allLabelFileNameLineCntList, log_prob_per_element_list)
        )

        if config.evaluation:
            # Prepraration for the hitory two-metric results
            twoMetricScore = read_file(results.system_path("train_summary"), "OutlierScores_metric.txt")
            ave_twoMetricScore = np.mean(np.array(twoMetricScore), axis=0).tolist()
            save_file(results.system_path("result_summary"), "PRF.txt",
                ["Average score of each univariate time series", "\n"], ",")
            save_file(results.system_path("result_summary"), "PRF.txt",
                ave_twoMetricScore+["\n"], ",", "a")
            save_file(results.system_path("result_summary"), "PRF.txt",
                ["Threshold", "F", "Precision", "Recall", "TP", "FP", "FN", "\n"], ",", "a")

            # get the sorted item each metric by change score
            twoMetricScoreList = cal_scoreChanges(log_prob_per_element_list,
                ave_twoMetricScore, log_prob_per_element_univariate_TS_list)
            MetricResult = session.run(SORT_UNIVARIATE_TS,
                feed_dict={SORT_UNIVARIATE_TS_INPUT: twoMetricScoreList})
            save_file(results.system_path("result_summary"), "MetricResult.txt",
                cat_List(allLabelFileNameLineCntList, MetricResult))

            # POT evalution
            POT_TH = pot_eval(
                read_file(results.system_path("train_summary"), "OutlierScores.txt", "float"), config.q, config.level
            )
            resultArray, outlierLabelfileNameLineCntList = cal_binaryResult(
                log_prob_per_element_list, POT_TH, time_indexs2, config.saveMetricInfo, allLabelFileNameLineCntList
            )
            evaluate(results, config.labelpath, resultArray, time_indexs2, POT_TH)

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()

    interpretation_hit_ratio(
        truth_filepath=config.interpret_filepath,
        prediction_filepath=os.path.join(config.result_dir, dirName, "result_summary", "MetricResult.txt")
    )


if __name__ == '__main__':
    main()
