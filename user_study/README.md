# Code for conducting the user study

The user study is a simple [Flask](https://flask.palletsprojects.com/en/3.0.x/) application that can be setup on any server.
We provide two versions of the study. One that trains a model on the server the study is running (`model_included.py`) and one that uses an externally trained model (`model_external.py`). For our experiments, we used `model_external.py` for our experiments. To run `model_included.py` you need some respective code from our experimental setup.

## Data

The database structure is provided in `annotation-curriculum.sql` and is connected via [SQLAlchemy](https://www.sqlalchemy.org/).
To run the application, you need to modify your name and password (lines 30-41) accordingly.

The tweet texts are not available by default. Instead, as stated in our ethics statement, you need to fetch them using the tweetIDs provided in the respective documents. The misconceptions are provided by the [CovidLies](https://github.com/ucinlp/covid19-data) corpus shared via an Apache 2 License.

After fetching the tweet texts, you need to complete the respective columns in the `.csv` files in the folder `annotation_task` .

## Running the instance

To run the study, first create a virtual environment (e.g., conda) and install the required packages:

    conda create --name=ac-study python=3.7
    conda activate ac-study
    pip install -r requirements.txt

Next, create a database (with an example user `admin` with the password `admin`):

    mysql -u admin -p
    CREATE DATABASE ac-study;

Then import the database structure 

    mysql -u admin -p ac-study < annotation-curriculum.sql

The application can be started via:

    python model_external.py

If you want to run the model locally, use:

    python model_included.py

Exporting data from the database can be done via:

    mysqldump -u admin -p ac-study --add-drop-table > annotation-curriculum.sql
