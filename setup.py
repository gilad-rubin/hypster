from setuptools import setup

setup(
	name='HyPSTER',
	version='0.0.2',
	packages=['hypster'],
	url='https://github.com/gilad-rubin/HyPSTER',
	license='',
	author='Gilad Rubin, Tal Peretz',
	author_email='',
	description='HyPSTER is a brand new Python package that helps you find compact and accurate ML Pipelines while staying light and efficient.',
	install_requires=[
		'category-encoders==2.0.0',
		'numpy>=1.17.1',
		'optuna>=0.14.0',
		'pandas>=0.24.0',
		'scikit-learn>=0.21.3',
		'scipy>=1.3.1',
		'xgboost==0.90',
	]
)
