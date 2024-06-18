# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run
Please fill this section as part of the assignment.



# Diamonds Price Prediction

This project focuses on building predictive models to estimate the price of diamonds based on various features using Linear Regression and XGBoost algorithms. The project includes data preparation, model training, evaluation, and hyperparameter tuning using Optuna.

## Table of Contents

1. [Setup](#setup)
2. [Directory structure](#directory-structure)
3. [Run procedure](#run-procedure)
4. [API requests](#api-requests)

## Setup
Ensure you have the following packages installed:

- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- optuna
- Flask

You can install these packages using pip, or using or using the requirements.txt


```bash
pip install pandas numpy scikit-learn xgboost joblib optuna Flask
pip install -r requirements.txt 
```

## Directory structure
Lets see what every script contains:
1. [constants.py](): contains some relative addresses of the folders used/created by the main.py file;
2. [function.py](): contains all the functions used by the main.py file, from reading data, to training models, to saving them;
3. [main.py](): main file for creating and saving the models used (requests 1 and 2 of the commissioned work). This is the first file to run in this project before moving on to the API services part;
4. [web_service.py](): contains the main flask app. To be run before requesting individual services. Note that this file also generates an additional file 'api_logs.db' which contains the list of requested services and their timestamp;
5. [predict_service.py](): predict value service;
6. [similar_service.py](): returns the list of gems similar to the one passed to it as input;
7. [getlogs_service.py](): show request log;

In particular, in the **models** folder N subfolders are generated (one for each type of model counted in the main), and a csv named **models_records.csv** where the performances and timestamp of the run model are saved;


## Run procedure

1. First of all, you need to run the **main.py** file. In this way, we will generate the implemented models and the corresponding folders in the models folder. Note that it is not necessary to generate the folders manually, as the save function in **function.py** generates them automatically. The same goes for the csv which records the performance and timestamp of the various models.

2. Once the various models have been generated, we can move on to creating the webservice, simply by running the **web_service.py** script. Note that for simplicity we chose to use the first XGBOOST model optimized for prediction (see **predict_service.py** script). Note that to call the individual services, the web_service.py file must be launched and kept active, while a new terminal must be generated, from which the individual services will be launched.

## API requests
Remembering to keep the web_service.py file active, we can move on to running our 3 services:

### Prediction service
Return the value predicted by the selected model. Open a new terminal and write:

```bash
python predict_service.py
```

### Similar service
Returns the list of gems similar to the one passed to it as input. It reads our base db and return a sample of it selecting only the gems close to our row passed with the service.
Open a new terminal and write:

```bash
python similar_service.py
```

### Get logs
Everytime a service is requested, it will be saved il the api_logs.db.
To get the whole file write in a new terminal:

```bash
python getlogs_service.py
```

