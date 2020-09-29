#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import sys
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from plotting_utils import *
from global_utils import *

import argparse
import pdb
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def getModel(params):
	sys.path.insert(0, global_params.py_models_path.format(params["model_name"]))
	if params["model_name"] == "esn":
		import esn as model
		return model.esn(params)
	elif params["model_name"] == "esn_parallel":
		import esn_parallel as model
		return model.esn_parallel(params)
	elif params["model_name"] == "rnn_statefull":
		import rnn_statefull as model
		return model.rnn_statefull(params)
	elif params["model_name"] == "rnn_statefull_parallel":
		import rnn_statefull_parallel as model
		return model.rnn_statefull_parallel(params)
	elif params["model_name"] == "mlp":
		import mlp as model
		return model.mlp(params)
	else:
		raise ValueError("model not found.")

def runModel(params_dict):
	params_dict["saving_path"] = global_params.saving_path.format(params_dict["system_name"])
	if params_dict["mode"] in ["train", "all"]:
		trainModel(params_dict)
	if params_dict["mode"] in ["test", "all"]:
		test_performance = testModel(params_dict)
	return test_performance

def black_box_function(params_dict, **kwargs):
	params_dict.update(kwargs)
	fval = runModel(params_dict)
	return fval

def trainModel(params_dict):
	model = getModel(params_dict)
	model.train()
	model.delete()
	del model
	return 0

def testModel(params_dict):
	model = getModel(params_dict)
	model.testing()
	test_performance = model.results_data['num_accurate_pred_005_avg_TEST'] - model.results_data['rmnse_avg_TEST'] - model.results_data['error_freq_TEST']
	model.delete()
	del model
	return test_performance


def defineParser():
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(help='Selection of the model.', dest='model_name')

	esn_parser = subparsers.add_parser("esn")
	esn_parser = getESNParser(esn_parser)

	esn_parallel_parser = subparsers.add_parser("esn_parallel")
	esn_parallel_parser = getESNParallelParser(esn_parallel_parser)

	rnn_statefull_parser = subparsers.add_parser("rnn_statefull")
	rnn_statefull_parser = getRNNStatefullParser(rnn_statefull_parser)

	rnn_statefull_parallel_parser = subparsers.add_parser("rnn_statefull_parallel")
	rnn_statefull_parallel_parser = getRNNStatefullParallelParser(rnn_statefull_parallel_parser)

	mlp_parser = subparsers.add_parser("mlp")
	mlp_parser = getMLPParser(mlp_parser)

	mlp_parallel_parser = subparsers.add_parser("mlp_parallel")
	mlp_parallel_parser = getMLPParallelParser(mlp_parallel_parser)

	return parser

def main():
	parser = defineParser()
	args = parser.parse_args()
	print(args.model_name)
	args_dict = args.__dict__

	# for key in args_dict:
		# print(key)

	# DEFINE PATHS AND DIRECTORIES
	args_dict["model_dir"] = global_params.model_dir
	args_dict["fig_dir"] = global_params.fig_dir
	args_dict["results_dir"] = global_params.results_dir
	args_dict["logfile_dir"] = global_params.logfile_dir
	args_dict["train_data_path"] = global_params.training_data_path.format(args.system_name, args.N)
	args_dict["test_data_path"] = global_params.testing_data_path.format(args.system_name, args.N)
	args_dict["worker_id"] = 0

	# print('Running', args_dict["saving_path"])
	pbounds = {'sigma_input': (0.5, 2), 'gamma': (1, 50)} #radius, regularization

	lambda_black_box_function = lambda **kwargs: black_box_function(args_dict, **kwargs)

	optimizer = BayesianOptimization(
	    f=lambda_black_box_function,
	    pbounds=pbounds,
	    random_state=1,
	)


	log_path = "./BayesOpt_log.json"
	# New optimizer is loaded with previously seen points
	try:
		load_logs(optimizer, logs=[log_path]);
	except:
		pdb.set_trace()
		print('unable to load old BayesOpt logs')
		pass
	logger = JSONLogger(path=log_path, reset=False)
	optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

	# optimizer.probe(
	#     params={"sigma_input": 10, "gamma": 30},
	#     lazy=True,
	# )
	#
	# optimizer.probe(
	#     params={"sigma_input": 10, "gamma": 15},
	#     lazy=True,
	# )
	#
	# optimizer.probe(
	#     params={"sigma_input": 7.5, "gamma": 30},
	#     lazy=True,
	# )

	optimizer.probe(
	    params={"sigma_input": 1, "gamma": 20},
	    lazy=True,
	)


	optimizer.probe(
	    params={"sigma_input": 0.5, "gamma": 20},
	    lazy=True,
	)


	optimizer.maximize(
	    init_points=1,
	    n_iter=3
	)



	for i, res in enumerate(optimizer.res):
	    print("Iteration {}: \n\t{}".format(i, res))

	print("MAX",optimizer.max)
	pdb.set_trace()

if __name__ == '__main__':
	main()
