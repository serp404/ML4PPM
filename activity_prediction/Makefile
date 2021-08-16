launch_catboost:
	export PYTHONPATH='.' && cat ../data/preprocessed_datasets/dataset_names | xargs -tI{} python ./boosting/launch.py {} helpdesk
	export PYTHONPATH='.' && cat ../data/preprocessed_datasets/dataset_names | xargs -tI{} python ./boosting/launch.py {} bpi12w
	export PYTHONPATH='.' && cat ../data/preprocessed_datasets/dataset_names | xargs -tI{} python ./boosting/launch.py {} bpi12
	export PYTHONPATH='.' && cat ../data/preprocessed_datasets/dataset_names | xargs -tI{} python ./boosting/launch.py {} bpi17
lint:
	isort .
	flake8
