# Disaster Response Pipeline Project

### Table of Contents
  1. [Installation](#Installation)
  2. [Project Motivation](#Project-Motivation)
  3. [File Descriptions](#File-Descriptions)
  4. [Results](#Results)
  5. [Licensing, Authors, and Acknowledgements](#licensing)

### Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The 
code should run with no issues using Python versions 3.x

1. Unzip the disaster_categories.csv.zip file in data directory into disaster_categories.csv
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

### Project Motivation
For this project, I was interestested in using disaster response data from Figure Eight to identify most important messages that need immediate response.

### File Descriptions
The full set of data files related to this course are owned by Figure Either [here](https://appen.com/resources/datasets/)

The project consists of 2 CSV files downloaded from Figure Eight. These are the data files. They are named as
  1. disaster_messages.csv
  2. disaster_categories.csv

The disaster_categories.csv is too big to be uploaded into github and hence has been compressed into disaster_categories.csv.zip file and uploaded. You should unzip the file into disaster_categories.csv before executing the code.

### Results

The application will present a form as shown below. 

![Homepage](images/Image01.png)

The users (people who respond to disaster events) can input any messages that they received in to this form and get a classification about the message. They can decide if the message is important or not based on these classification tags.

![Response](images/Image02.png)

### Licensing, Authors, and Acknowledgements<a name="licensing"></a>
The data for this analysis was downloaded from [Kaggle](https://www.kaggle.com/airbnb/seattle/data). Otherwise, feel free to use the code here as you would like!



