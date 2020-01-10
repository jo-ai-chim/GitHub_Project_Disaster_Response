# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Project Motivation](#motivation)
4. [Files](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

To run the code in this project beside the standard libraries already included in the standard Anaconda installation (numpy, pandas, matplotlib, datetime, sklearn, time and seaborn) you need to install 

- plotly library the according documentation can be found [here](https://plot.ly/).
- textblob library the according documentation can be found [here](https://textblob.readthedocs.io/en/dev/).

The code should run with no issues using Python versions 3.*.

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/disasterresponse_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/

## Project Motivation <a name="motivation"></a>

Goal of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## Files <a name="files"></a>

The projects contains 8 files beside this readme.

Data files:

- disaster_categories.csv in data folder - Input Data for training and testing the model
- disaster_messages.csv in data folder - Input Data for training and testing the model
- disasterresponse_model.pkl in models folder - file to save the model in

code files:

- process_data.py in data folder - contains the etl pipeline to read in the input data, transform it and save it in the database
- train_classifier.py in models folder - set's up and evaluates the model and the stores it to a pickle file for the webpage to use it 
- run.py in app folder - used to run the webapp
- go.html in app folder - html file to display the results
- master.html in app folder - html file for the main page

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

This project is based on the template for the project "Disaster Response Pipelines" within the Data Scientist for Enterprise Nano Degree Programm.
