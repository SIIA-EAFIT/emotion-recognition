# emotion-recognition

A [Flask](http://flask.pocoo.org/) app to recognize the emotion of your (anyone's) face. We train the models using the data from the [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) at [Kaggle](https://www.kaggle.com/).

## Versions

In this project we try out different approaches for image classification. Each approach will be located in a different branch from the master branch, in this way everyone can contribute an approach without making any conflicts. All branches must contain the followinf files:

* *flask_app.py*: the flask application.
* *README.md*: this file, the only difference between branches should be in the *approach description section*.
* *requirements.txt*: a requirements file with all dependencies of the approach.

The main branch of this repository should always correspond to the approach with the highest F1-score

### Ranking

| User     | Branch | F1-score (test) |
| ---      | ---      | ---      |
| [diegoxfx](https://github.com/diegoxfx) |          |          |
| [TheBaxes](https://github.com/TheBaxes) |          |          |
| [srcolinas](https://github.com/srcolinas) |          |          |
| ...      |  ...     | ...       |


### Approach of current version

There is no model yet.

## Project structure

```bash
|-data/
|    __init__.py
|    ...
|-model/
|    __init__.py
|    ...
|-flask_app.py
|-README.md
|-requirements.txt
```
The following describes the responsibility of each component in the above folder structure:
* *tests*: include notebooks to show how the functions and classes defined in the [data](/data/) and [model](/model/) work
* *data*: download and preprocess the data
* *model*: train, test and infer
* *flask_app.py*: the web application
* *README.md*: to describe the project (this file)
* *requirements.txt*: store all necessary libraries for this branch to work

Make sure you keep unnecessary folder and files inside the [.gitignore](/.gitignore) file

## Set up the virtual environment

First make sure you have a working installation of [Pyhton 3.6 or later](https://www.python.org/downloads/) and you have [Virtualenv](https://virtualenv.pypa.io/en/stable/) installed.

issue the following command on the terminal to create the virtual environment `virtualenv venv`, then do either `source venv/bin/activate` (on Linux) or `venv\Scripts\activate` (on Windows), followed by `pip intall -r requirements.txt`.

## Start the server

`python flask_app.py`



