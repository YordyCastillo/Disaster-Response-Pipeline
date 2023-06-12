# Disaster Response Pipeline Project

This project is part of the Disaster Response Pipeline project from Udacity's Data Science Nanodegree program. The goal of the project is to build a web application that can classify messages related to disasters into different categories, allowing for quick identification and response during emergencies.

The project consists of three main components:
1. ETL Pipeline: This pipeline is responsible for cleaning and preprocessing the message and category data, and storing it in a SQLite database for later use.
2. ML Pipeline: This pipeline trains a machine learning model on the preprocessed data to classify messages into different categories. The trained model is then saved for future use.
3. Web Application: The web application provides a user interface where users can enter messages and get them classified into relevant categories. It also displays visualizations based on the data extracted from the SQLite database.

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model:

    - To run the ETL pipeline that cleans the data and stores it in the database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run the ML pipeline that trains the classifier and saves the model:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

   **Note:** If you encounter any issues related to scikit-learn version compatibility, you may need to upgrade scikit-learn to version 0.24.2 or higher. You can upgrade scikit-learn by running `pip install --upgrade scikit-learn`.

2. Go to the `app` directory:
   `cd app`

3. Run the web app:
   `python run.py`

4. Open your web browser and enter the following URL:
   `http://localhost:3000`

5. Use the web application to enter messages and get them classified into relevant categories. Explore the visualizations provided based on the data extracted from the SQLite database.

## File Descriptions:

- `app` directory: Contains the files related to the web application.
  - `run.py`: Starts the Flask web server and defines the routes for the web app.
  - `templates` directory: Contains the HTML templates for the web pages.
    - `master.html`: Main page template that displays the visuals and user input form.
    - `go.html`: Page template that displays the classification results.

- `data` directory: Contains the data files and ETL pipeline script.
  - `disaster_messages.csv`: CSV file containing the raw message data.
  - `disaster_categories.csv`: CSV file containing the raw category data.
  - `process_data.py`: ETL pipeline script that cleans and preprocesses the data.

- `models` directory: Contains the ML pipeline script.
  - `train_classifier.py`: ML pipeline script that trains the classifier and saves the model.

## Dependencies:

The project has the following dependencies:

- Python 3.x
- Data manipulation and analysis libraries: NumPy, Pandas
- Machine learning libraries: Scikit-learn (version 0.24.2 or higher)
- Natural language processing libraries: NLTK
- SQLite database library: SQLAlchemy
- Web framework: Flask
- Data visualization libraries: Plotly

You can install the required dependencies by running:
`pip install -r requirements.txt`

## Acknowledgements:

The project data was provided by Figure Eight (now Appen) and is part of the Data Science Nanodegree program at Udacity. Special thanks to Udacity for providing the project resources and guidelines.

## Graphs From Web Page

![Image Description](https://github.com/YordyCastillo/Disaster-Response-Pipeline/raw/main/newplot%20(1).png)
![Image Description](https://github.com/YordyCastillo/Disaster-Response-Pipeline/raw/main/newplot%20(2).png)
![Image Description](https://github.com/YordyCastillo/Disaster-Response-Pipeline/raw/main/newplot%20(3).png)


