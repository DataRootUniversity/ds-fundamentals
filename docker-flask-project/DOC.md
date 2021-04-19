# RESTful APIs Project Guide

*Hello and welcome to the **RESTful APIs Project** tutorial! Today you'll learn how to **structure** your project and **develop** your own **web API**.*

There are lots of different ways to structure your RESTful API. Here we have gathered the **best practices** of project organization. During the tutorial, we will provide an example of a well-structured project to show you how to make your code **organized** and **effective**.

# Table Of Contents

- [Intro](#intro)
- [Before we start](#before-we-start)
- [Folder structure](#folder-structure)
- [Main Components](#main-components)
	- [Setting up your database](#setting-up-your-database)
	- [Models](#models)
    - [Controllers](#controllers)
    - [Core](#core)
- [Test your app](#test-your-app)
- [Dockerize your app](#dockerize-your-app)

## Intro

In this tutorial, we're going to implement a web **API** (**A**pplication **P**rogramming **I**nterface) using **REST** (**RE**presentational **S**tateless **T**ransfer) approach and based on **MVC** (**M**odel, **V**iew, **C**ontroller) architectural pattern. Our service will allow users to interact with the database and make basic **CRUD** (**C**reate, **R**ead, **U**pdate, **D**elete) operations. We will use **PostgreSQL** as a database because of its robustness and high performance.

This template is based on best structuring patterns, so it can be adapted for any other similar project.

## Before we start 
Now you will start building you own API, that contains 2 entities (Actors and Movies) and their relations. 
The main purpose of this project is to teach you how to create your own RESTful API and Dockerize it. So the task itself is not a purpose, just exercise. 

**The project plan is as follows:**
1. Create database and connect to it
2. Create Models of Actors and Movies 
3. Create methods for interaction with Models (`commit()`, `create()`, `update()` etc.)
4. Create methods for correct processing requests and handling errors (controllers) (`get_actor_by_id()`, `add_movie()` etc.)
5. Create routes for app corresponding to our controllers
6. Test your app
7. Dockerize your app

## Folder structure

As mentioned earlier, we'll use the **MVC** design pattern to separate our project into the logical modules: **Model**, **View**, **Controller**:

- **Model** is for retrieving and managing of all necessary data from a database.
- **View** serves for any output representation or data rendering from the model into the user interface.
- **Controller** accepts input and converts it to commands for model/view.
In this project, we won't implement the **View** component, cause it's not necessary for now.

Here is our folder structure where all main components are mentioned. For now we will only define them, but later we'll consider them in details:

```
app
    ├──  models                 - this folder contains all database models.
    │   ├── __init__.py         - init file for the package.
    │   ├── base.py             - this file contains class Model which handles all data-management operations. Actor and Movie classes will be inherited from it.
    │   ├── actor.py            - Actor entity model.
    │   ├── movie.py            - Movie entity model.
    |   └── relations.py        - association table for Actor and Movie entities. 
    │
    │
    ├── controllers             - this folder contains all commands operations handlers.
    │   ├── actor.py            - handlers for operations related to the Actor entity.
    │   ├── movie.py            - handlers for operations related to the Movie entity.
    │   └── parse_request.py    - this file contains a function which parses request data and converts it to a convenient format.
    │   
    │   
    ├── settings               - here you can store different constant values, connection parameters, etc.
    │   └── constants.py        -  multiple constants storage for their convenient usage.
    │ 
    │ 
    ├── core                    - folder, which contains core application components.
    │   ├── __init__.py         - initializing our app and DB.
    │   └── routes.py           - application routes (predefined commands).
    │ 
    │ 
    ├── run.py                  - application run file.
    |
    ├── Dockerfile				- commands used for Dockerization
    |
    └── requirements.txt		- list of libraries used for Dockerization
```
Now it's time for you to define the structure by carrying it to your local machine.

## Main components

**Important Notes:**

1. In order to complete this project, it's better for you to have installed [**PyCharm**](https://www.jetbrains.com/pycharm/download/)
2. If you are student of any university cou can apply for [**JetBrains Free Educational Licenses**](https://www.jetbrains.com/community/education/#students) and get **PyCharm Professional** for free (only for the period of study)
3. Also, you'll need to install [**Docker**](https://www.docker.com).

### Setting up your database

Firstly we need to set up a database we will work with. In order to implement the project, you need to [install **Postgres**](https://www.postgresql.org/download/) locally to your machine. 

**To create your database:**
1. Open SQL Shell 
2. Login, setting `Server` as `localhost` and `Port` as `5432` (usually they are defaults)  
3. `CREATE DATABASE test_db;` 
4. `CREATE USER test_user WITH ENCRYPTED PASSWORD 'password';`
5. `GRANT ALL PRIVILEGES ON DATABASE test_db TO test_user;`

### Models

There are two entities: **Actor** and **Movie**. One actor can star in multiple movies, the movie's cast can consist of multiple actors. So the relation is **many-to-many**.
You can see all properties in the schema below:

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/db-diagram.png?raw=true">
</div>


We need to implement **models** of these **entities**.
As you can see, some more constants to appear here, so let's put them all to the `settings/constants.py`:
```python
import os
# db connection URL (In order to submit your project do NOT change this value!!!)
DB_URL = os.environ['DB_URL']                                                                     
# entities properties
ACTOR_FIELDS = ['id', 'name', 'gender', 'date_of_birth']
MOVIE_FIELDS = ['id', 'name', 'year', 'genre']

# date of birth format
DATE_FORMAT = '%d.%m.%Y'
```
### Little explanation:
In order to connect your app to your database you need to get it's url.
When Bot validates your lab, it creates environmental variable `DB_URL`. 
So, when you will debug your project you should create it the same way.
Put this command into the terminal in **PyCharm**:
For Windows:
```
set DB_URL=postgresql+psycopg2://test_user:password@localhost:5432/test_db
```
For macOS and UNIX:
```
export DB_URL=postgresql+psycopg2://test_user:password@localhost:5432/test_db
```

So now to run your python file with respect to environmental variables you should run it from **PyCharm** terminal in such way:
```
python run.py
```
###  Associations 
We'll use [`SQLAlchemy`](https://www.sqlalchemy.org/) toolkit to manipulate all stuff related to the DataBase (defining models, initialization of the DB, etc).

Let's define an **association** table, keeping in mind that our **entities** have **relations** between each other.

> **Important note:**
To start working with the models we need to initialize the `DB` object. Go to the `core/__init__.py` and define it:
```python
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()
```
### Models
Now we need to define the relations between entities. Look through the [documentation](https://docs.sqlalchemy.org/en/13/orm/basic_relationships.html) and then move to the `models/relations.py` and implement an **association table** following the instructions below:
```python
from core import db
from sqlalchemy import Table, Column, Integer, ForeignKey

# Table name -> 'association'
# Columns: 'actor_id' -> db.Integer, db.ForeignKey -> 'actors.id', primary_key = True
#          'movie_id' -> db.Integer, db.ForeignKey -> 'movies.id', primary_key = True
association =
```

Let's try to create our first model `Actor`. Check out [documentation](https://flask-sqlalchemy.palletsprojects.com/en/2.x/models/), then open `models/actor.py` and complete the model's class with defining entities properties:
```python
from datetime import datetime as dt

from core import db
from models.relations import association

class Actor(db.Model):
    __tablename__ = 'actors'

    # id -> integer, primary key
    id =
    # name -> string, size 50, unique, not nullable
    name =
    # gender -> string, size 11
    gender =
    # date_of_birth -> date
    date_of_birth =

    # Use `db.relationship` method to define the Actor's relationship with Movie.
    # Set `backref` as 'cast', uselist=True
    # Set `secondary` as 'association'
    movies =

    def __repr__(self):
        return '<Actor {}>'.format(self.name)
```
And the same procedure for the `Movie` model. Open `models/movie.py` and complete the following code:
```python
from datetime import datetime as dt

from core import db
from models.relations import association


class Movie(db.Model):
    __tablename__ = 'movies'

    # id -> integer, primary key
    id = 
    # name -> string, size 50, unique, not nullable
    name =  
    # year -> integer
    year = 
    # genre -> string, size 20
    genre = 

    # Use `db.relationship` method to define the Movie's relationship with Actor.
    # Set `backref` as 'filmography', uselist=True
    # Set `secondary` as 'association'
    actors = 

    def __repr__(self):
        return '<Movie {}>'.format(self.name)
```
Upon completing the `Actor` and `Movie` model, let's test them.

> **Important note:**
Remember that it's better to test your code. You can use **the terminal** or **Jupyter Notebook** for this. In this tutorial, we use **Jupyter Notebook**

> **PYTHONPATH**
Don't forget to add your directory to `PYTHONPATH` to allow importing from your project files:
`set PYTHONPATH='${PYTHONPATH}:/path/to/your/project`
Or
`export PYTHONPATH='${PYTHONPATH}:/path/to/your/project`

In order to check our models, we need to get `db` instance from `core` and initialize some dummy Flask `app`:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime as dt

from settings.constants import DB_URL
from core import db
from models.actors import Actor  
from models.movie import Movie


app = Flask(__name__, instance_relative_config=False)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # silence the deprecation warning

db.init_app(app)
```
Now let's test how it works adding some data to the table. We need to use `app.app_context()` to do this:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime as dt

from settings.constants import DB_URL
from core import db
from models.actors import Actor  
from models.movie import Movie


app = Flask(__name__, instance_relative_config=False)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # silence the deprecation warning

db.init_app(app)

data = {'name': 'Megan Fox', 'gender': 'female', 'date_of_birth': dt.strptime('16.05.1986', '%d.%m.%Y').date()}

with app.app_context():
    db.create_all()
    obj = Actor(**data)
    db.session.add(obj)
    db.session.commit()
    db.session.refresh(obj)
    print(obj)
    print(obj.__dict__)
```
Here is an example of the output:

<div align="center">
    <img align="center" width="706" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/add-actor-raw.png?raw=true">
</div>

Also, we recommend that you look through the **documentation** and **test other operations** (delete record, add relations, remove relations, etc) in the same way.

Great! Our models are done. It's time to move on to the more interesting part - **handling of data manipulating operations.**
As you can see, operations are the same for both of our models. We need to be able to:

- **add/update/delete** record
- **add/update/delete** relation

However, we need to avoid rewriting the same code two times for both models. So, let's use the power of metaprogramming and implement class `Model` where we will have the needed methods as `@classmethods`.
Move to `models/base.py`, follow instructions, and implement needed methods:
```python
from core import db


def commit(obj):
    """
    Function for convenient commit
    """
    db.session.add(obj)
    db.session.commit()
    db.session.refresh(obj)
    return obj


class Model(object):
    @classmethod
    def create(cls, **kwargs):
        """
        Create new record

        cls: class
        kwargs: dict with object parameters
        """
        obj = cls(**kwargs)
        return commit(obj)

    @classmethod
    def update(cls, row_id, **kwargs):
        """
        Update record by id

        cls: class
        row_id: record id
        kwargs: dict with object parameters
        """
        obj = 
        return commit(obj)
    
    @classmethod
    def delete(cls, row_id):
        """
        Delete record by id

        cls: class
        row_id: record id
        return: int (1 if deleted else 0)
        """
        obj = 
        return obj
    
    @classmethod
    def add_relation(cls, row_id, rel_obj):  
        """
        Add relation to object

        cls: class
        row_id: record id
        rel_obj: related object
        """      
        obj = cls.query.filter_by(id=row_id).first()
        if cls.__name__ == 'Actor':
            obj.filmography.append(rel_obj)
        elif cls.__name__ == 'Movie':
            obj.cast.append(rel_obj)
        return commit(obj)
            
    @classmethod
    def remove_relation(cls, row_id, rel_obj):
        """
        Remove certain relation

        cls: class
        row_id: record id
        rel_obj: related object
        """
        obj = 
        return commit(obj)

    @classmethod
    def clear_relations(cls, row_id):
        """
        Remove all relations by id

        cls: class
        row_id: record id
        """
        obj = 
        return commit(obj)
```
Awesome! Now we can **inherit** our **models** from this class and use its methods as **class methods** of `Actor` and `Movie`! Let's add this feature to `Actor`:
```python
    from datetime import datetime as dt
    
    from core import db
	from models.base import Model  
	from models.relations import association
    
    
    class Actor(Model, db.Model):
        . . .
```

and `Movie`:
```python
    from datetime import datetime as dt
    
    from core import db
	from models.base import Model  
	from models.relations import association
    
    
    class Movie(Model, db.Model):
        . . .
```
Let's test this by repeating the previous test operation but using class method `create`:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime as dt
from sqlalchemy import inspect

from settings.constants import DB_URL
from core import db
from models.actors import Actor  
from models.movie import Movie

data = {'name': 'Megan Fox', 'gender': 'female', 'date_of_birth': dt.strptime('16.05.1986', '%d.%m.%Y').date()}

app = Flask(__name__, instance_relative_config=False)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # silence the deprecation warning

db.init_app(app)

with app.app_context():
    db.create_all()
    obj = Actor.create(**data)
    print(obj)
    print(obj.__dict__)
```
Here is an example of the output:

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/add-actor-method.png?raw=true">
</div>

**Now you should test every operation.** Here is an example of methods usage:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime as dt
from sqlalchemy import inspect

from settings.constants import DB_URL
from core import db
from models.actors import Actor  
from models.movie import Movie


data_actor = {'name': 'Megan Fox', 'gender': 'female', 'date_of_birth': dt.strptime('16.05.1986', '%d.%m.%Y').date()}
data_actor_upd = {'name': 'Not Megan Fox', 'gender': 'male', 'date_of_birth': dt.strptime('16.05.2000', '%d.%m.%Y').date()}

data_movie = {'name': 'Transformers', 'genre': 'action', 'year': 2007}
data_movie_upd = {'name': 'Teenage Mutant Ninja Turtles', 'genre': 'bad movie', 'year': 2014}

app = Flask(__name__, instance_relative_config=False)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # silence the deprecation warning

db.init_app(app)

with app.app_context():
    db.create_all()
    actor = Actor.create(**data_actor)
    print('created actor:', actor.__dict__, '\n')

    movie = Movie.create(**data_movie)
    print('created movie:', movie.__dict__, '\n')

    upd_actor = Actor.update(1, **data_actor_upd)
    print('updated actor:', upd_actor.__dict__, '\n')

    upd_movie = Movie.update(1, **data_movie_upd)
    print('updated movie:', upd_movie.__dict__, '\n')

    add_rels_actor = Actor.add_relation(1, upd_movie)
    movie_2 = Movie.create(**data_movie)
    add_more_rels_actor = Actor.add_relation(1, movie_2)
    print('relations list:', add_more_rels_actor.filmography, '\n')

    clear_rels_actor = Actor.clear_relations(1)
    print('all relations cleared:', clear_rels_actor.filmography, '\n')

    del_actor = Actor.delete(1)
    print('actor deleted:', del_actor)
```
The output must be as following:
```
created actor: {'_sa_instance_state': <sqlalchemy.orm.state.InstanceState object at 0x109b19590>, 'gender': 'female', 'id': 1, 'date_of_birth': datetime.date(1986, 5, 16), 'name': 'Megan Fox'} 

created movie: {'_sa_instance_state': <sqlalchemy.orm.state.InstanceState object at 0x109f94650>, 'genre': 'action', 'name': 'Transformers', 'year': 2007, 'id': 1} 

updated actor: {'_sa_instance_state': <sqlalchemy.orm.state.InstanceState object at 0x109b19590>, 'gender': 'male', 'id': 1, 'date_of_birth': datetime.date(2000, 5, 16), 'name': 'Not Megan Fox'} 

updated movie: {'_sa_instance_state': <sqlalchemy.orm.state.InstanceState object at 0x109f94650>, 'genre': 'bad movie', 'name': 'Teenage Mutant Ninja Turtles', 'year': 2014, 'id': 1} 

relations list: [<Movie Teenage Mutant Ninja Turtles>, <Movie Transformers>] 

all relations cleared: [] 

actor deleted: 1
```
Congrats, **Models** module is done!

### Controllers

Let's move further and implement the **Controllers** module.

As we already know, **controllers** can be named as **commands handlers**. In our case, we'll handle commands related to the previously defined models' operations. So, the controller must **process** input data, **pass** it through the operation, handling all possible exceptions and return some output.
It's better to split `Actor` and `Movie` controllers to separate files for convenience.
Data will come to the controller from **API request**, so firstly let's define a function for parsing request data.
Open `controllers/parse_request.py` and implement a function that **converts request data to `dict`**:
> **Important note:**
Make sure that you accept and parse data from the correct request field. The data will be encoded as **application/x-www-form-urlencoded**!
```python
from flask import request


def get_request_data():
    """
    Get keys & values from request (Note that this method should parse requests with content type "application/x-www-form-urlencoded")
    """

    return data
```
Now we can move to `controllers/actor.py` and implement operations handlers which will deal with the requests for the `Actor` model.
Main purpose of `controllers` is not only define application operations, but handle incorrect input requests.
Now you need to implement a few operations with appropriate error handlers:
- **`get_all_actors()`**
- **`get_actor_by_id():`**
  - id should be specified
  - id should be integer
  - Such actor id record should exist
- **`add_actor():`**
   - Inputted fields should exist
   - Among input fields should be date of birth 
   - Date of birth should be in format `DATE_FORMAT` (find it in `settings/constants`)
- **`update_actor():`**
	- id should be specified
	- id should be integer
	- Such actor id record should exist
   - Inputted field should exist
   - Among inputs fields should be date of birth 
   - Date of birth should be in format `DATE_FORMAT` (find it in `settings/constants`)
- **`delete_actor():`**
  - id should be specified
  - id should be integer
  - Such actor id record should exist
- **`actor_add_relation():`**
  - ids for actor and movie should be specified
  - ids should be integer
  - Such actor and movie ids record should exist
- **`actor_clear_relations():`**
  - ids for actor and movie should be specified
  - ids should be integer
  - Such actor and movie ids record should exist 

Use [`flask.make_response`](https://kite.com/python/docs/flask.make_response) to construct correctly formatted responses. For **correct** response use code `200`, and `400` for **bad requests**.
Here is a **template** for you to implement needed operations handlers (with a few completed examples as a bonus)
```python
from flask import jsonify, make_response

from datetime import datetime as dt
from ast import literal_eval

from models.actors import Actor  
from models.movie import Movie
from settings.constants import ACTOR_FIELDS     # to make response pretty
from .parse_request import get_request_data


def get_all_actors():
    """
    Get list of all records
    """  
    all_actors = Actor.query.all()
    actors = []
    for actor in all_actors:
        act = {k: v for k, v in actor.__dict__.items() if k in ACTOR_FIELDS}
        actors.append(act)
    return make_response(jsonify(actors), 200) 

  
def get_actor_by_id():
    """
    Get record by id
    """
    data = get_request_data()
    if 'id' in data.keys():
        try:
            row_id = int(data['id'])
        except:
            err = 'Id must be integer'
            return make_response(jsonify(error=err), 400) 

        obj = Actor.query.filter_by(id=row_id).first()
        try:
            actor = {k: v for k, v in obj.__dict__.items() if k in ACTOR_FIELDS}
        except:
            err = 'Record with such id does not exist'
            return make_response(jsonify(error=err), 400) 

        return make_response(jsonify(actor), 200)

    else:
        err = 'No id specified'
        return make_response(jsonify(error=err), 400) 


def add_actor():
    """
    Add new actor
    """
    data = get_request_data()
    ### YOUR CODE HERE ###

    # use this for 200 response code
    new_record = 
    new_actor = {k: v for k, v in new_record.__dict__.items() if k in ACTOR_FIELDS}
    return make_response(jsonify(new_actor), 200)
    ### END CODE HERE ###


def update_actor():
    """
    Update actor record by id
    """
    data = get_request_data()
    ### YOUR CODE HERE ###

    # use this for 200 response code
    upd_record = 
    upd_actor = {k: v for k, v in upd_record.__dict__.items() if k in ACTOR_FIELDS}
    return make_response(jsonify(upd_actor), 200)
    ### END CODE HERE ###

def delete_actor():
    """
    Delete actor by id
    """
    data = get_request_data()
    ### YOUR CODE HERE ###

    # use this for 200 response code
    msg = 'Record successfully deleted'
    return make_response(jsonify(message=msg), 200)
    ### END CODE HERE ###


def actor_add_relation():
    """
    Add a movie to actor's filmography
    """
    data = get_request_data()
    ### YOUR CODE HERE ###

    # use this for 200 response code
    actor =     # add relation here
    rel_actor = {k: v for k, v in actor.__dict__.items() if k in ACTOR_FIELDS}
    rel_actor['filmography'] = str(actor.filmography)
    return make_response(jsonify(rel_actor), 200)
    ### END CODE HERE ###


def actor_clear_relations():
    """
    Clear all relations by id
    """
    data = get_request_data()
    ### YOUR CODE HERE ###

    # use this for 200 response code
    actor =     # clear relations here
    rel_actor = {k: v for k, v in actor.__dict__.items() if k in ACTOR_FIELDS}
    rel_actor['filmography'] = str(actor.filmography)
    return make_response(jsonify(rel_actor), 200)
    ### END CODE HERE ###
```
> **Important note**
Don't forget to **test** your functions! You can just **comment out** the line with `get_request_data` call and pass a `dict` to the function like this:
```python
def get_actor_by_id(data):
    """
    Get record by id
    """
    # data = get_request_data()
    if 'id' in data.keys():
        try:
            row_id = int(data['id'])
        except:
            err = 'Id must be integer'
            return make_response(jsonify(error=err), 400) 

        obj = Actor.query.filter_by(id=row_id).first()
        try:
            actor = {k: v for k, v in obj.__dict__.items() if k in ACTOR_FIELDS}
        except:
            err = 'Record with such id does not exist'
            return make_response(jsonify(error=err), 400) 

        return make_response(jsonify(actor), 200)

    else:
        err = 'No id specified'
        return make_response(jsonify(error=err), 400) 

        obj = Actor.query.filter_by(id=row_id).first()
        try:
            actor = {k: v for k, v in obj.__dict__.items() if k in ACTOR_FIELDS}
        except:
            err = 'Record with such id does not exist'
            return make_response(jsonify(error=err), 400) 

        return make_response(jsonify(actor), 200)

```
Now it's time to implement `Movie` operations in the same way:
- **`get_all_movies()`**
- **`get_movie_by_id():`**
  - id should be specified
  - id should be integer
  - Such movie id record should exist
- **`add_movie():`**
   - Inputted fields should exist
- **`update_movie():`**
	- id should be specified
	- id should be integer
	- Such movie id record should exist
   - Inputted fields should exist
- **`delete_movie():`**
  - id should be specified
  - id should be integer
  - Such movie id record should exist
- **`movie_add_relation():`**
  - ids for actor and movie should be specified
  - ids should be integer
  - Such actor and movie ids record should exist
- **`actor_clear_relations():`**
  - ids for actor and movie should be specified
  - ids should be integer
  - Such actor and movie ids record should exist 
```python
from flask import jsonify, make_response

from ast import literal_eval

from models.actors import Actor  
from models.movie import Movie
from settings.constants import MOVIE_FIELDS
from .parse_request import get_request_data


def get_all_movies():
    """
    Get list of all records
    """


def get_movie_by_id():
    """
    Get record by id
    """
    

def add_movie():
    """
    Add new movie
    """


def update_movie():
    """
    Update movie record by id
    """


def delete_movie():
    """
    Delete movie by id
    """
    

def movie_add_relation():
    """
    Add actor to movie's cast
    """


def movie_clear_relations():
    """
    Clear all relations by id
    """
```
Perfect! The **Controllers** module is done!

### Core

We are close to finishing!

Now it's time to deal with the `app` part. To do this, firstly we need to write some `routes` corresponding to our controllers.
We'll use the following routes methods:

- `GET` to retrieve information from the source
- `POST` to send data to the source
- `PUT` to replace (update) data
- `DELETE` to remove data

We'll have following routes:

- `/api/actors` to get the list of all actors from the DB, method: `GET`
- `/api/movies`, to get the list of all movies from the DB, method: `GET`
- `/api/actor`, to manipulate with the actors records, methods:
    - `GET`: get actor by id
    - `POST`: add new actor, `body` can include:
        - `name, gender, date_of_birth`
    - `PUT`: update actor, `body` can include:
        - `id`, `name, gender, date_of_birth`
    - `DELETE`: remove actor, `body`:
        - `id`
- `/api/movie`, to manipulate with the movies records, methods:
    - `GET`: get movie by id
    - `POST`: add new movie, `body` can include:
        - `name, year, genre`
    - `PUT`: update movie, `body` can include:
        - `id`, `name, year, genre`
    - `DELETE`: remove movie, `body`:
        - `id`
- `/api/actor-relations` to manipulate with actor's relations, methods:
    - `PUT`: add relations, `body`:
        - `id, relation_id`
    - `DELETE`: delete relations, `body`:
        - `id`
- `/api/movie-relations` to manipulate with movies's relations, methods:
    - `PUT`: add relations, `body`:
        - `id, relation_id`
    - `DELETE`: delete relations, `body`:
        - `id`

Move to `core/routes.py` and write routes using previously implemented `controllers`:
```python
from flask import Flask, request
from flask import current_app as app

from controllers.actor import *
from controllers.movie import *


@app.route('/api/actors', methods=['GET'])  
def actors():  
    """  
 Get all actors in db """  
 return get_all_actors()  
  
  
@app.route('/api/movies', methods=['GET'])  
def movies():  
    """  
 Get all movies in db """  
 return get_all_movies()  
  
  
@app.route('/api/actor', methods=['GET', 'POST', 'PUT', 'DELETE'])  
def actor():  
    ...

@app.route('/api/movie', methods=['GET', 'POST', 'PUT', 'DELETE'])  
def movie():  
	...
  
  
@app.route('/api/actor-relations', methods=['PUT', 'DELETE'])  
def actor_relation():  
	... 
  
  
@app.route('/api/movie-relations', methods=['PUT', 'DELETE'])  
def movie_relation():  
	...
```
Then move to `core/__init__.py` and write a function to construct your application:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from settings.constants import DB_URL


db = SQLAlchemy()

def create_app():
    """Construct the core application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # silence the deprecation warning
    
    db.init_app(app)

    with app.app_context():
        # Imports
        from . import routes

        # Create tables for our models
        db.create_all()

            return app
```
And the last what we need to do is to create an application run file. Open the `run.py` and write the following lines:
```python
from core import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
```
## **Test your app**

In order to test the application, you need to run `run.py` and **debug** in case of errors.
Then you can use some `GUI` for sending requests like [Postman](https://www.getpostman.com/downloads/) or `curl` requests in Python.

In case you are using Postman:
1. Create new Workspace
2. Put into the URL box `http://0.0.0.0:8000/route-you-want-to-test` 
3. Choose method (GET, POST, PUT, DELETE etc.)
4. Choose "Body" and "form-data"
5. Put into the table keys and values you want to send to your app
6. Click "Send"

>**Example:**
>Method: `POST` 
	URL: `http://0.0.0.0:8000/api/actor `
	Body: `{"name": "Megan Fox", "date_of_birth": "01.01.1970", "gender": "female"}`
<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/add-actor.png?raw=true">
</div>

Below will be examples of the correct operation of the app. 
You don't have to test them all manually, but Bot will validate your app in similar way. 

Here are examples of **correct** requests with **`200`** response status code:

1. **Method:** `GET` 
	**URL:** `http://0.0.0.0:8000/api/actors`
	**Body:**`{}`	
	
2. **Method:** `GET` 
	**URL:** `http://0.0.0.0:8000/api/movies`
	**Body:**`{}`	
3. **Method:** `POST` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"name": "Megan Fox", "date_of_birth": "01.01.1970", "gender": "female"}`
4. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"id": 1, "name": "Megan Fox Fox"}`
5. **Method:** `GET` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"id": 1}`
6. **Method:** `DELETE` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"id": 1}`
7. **Method:** `POST` 
	**URL:** `http://0.0.0.0:8000/api/movie`
	**Body:** `{"name": "Fight Club", "year": "1999", "genre": "drama"}`
8. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/movie`
	**Body:** `{"id": 1, "name": "Fight Club Club"}`
9. **Method:** `GET` 
	**URL:** `http://0.0.0.0:8000/api/movie`
	**Body:**`{"id": 1}`
10. **Method:** `DELETE` 
	**URL:** `http://0.0.0.0:8000/api/movie`
	**Body:** `{"id": 1}`
11. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/actor-relations`
	**Body:** `{"id": 1, "relation_id": "1"}`
12. **Method:** `DELETE` 
	**URL:** `http://0.0.0.0:8000/api/actor-relations`
	**Body:** `{"id": 1}`
13. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/movie-relations`
	**Body:** `{"id": 1, "relation_id": "1"}`
14. **Method:** `DELETE` 
	**URL:** `http://0.0.0.0:8000/api/movie-relations`
	**Body:** `{"id": 1}`



And here are examples of **bad request** requests with **`400`** response status code (Error message doesn't matter):

1. **Method:** `POST` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"name": "Megan Fox", "date_of_birth": "02.20.1970", "gender": "female"}`
	**Note:** Wrong date format (20th month specified)
2. **Method:** `POST` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"name": "Megan Fox", "date_of_birth": "02.20.1970", "gender": "female", "height": "180"}`
	**Note:** Wrong input field ("height")
3.  **Method:** `POST` 
	**URL:** `http://0.0.0.0:8000/api/movie`
	**Body:** `{"name": "Fight Club", "year": "1999", "genre": "drama", "length": "120"}`
	**Note:** Wrong input field ("length")
	
4. **Method:** `GET`  or `DELETE`
	**URL:** `http://0.0.0.0:8000/api/actor` or `http://0.0.0.0:8000/api/movie`
	**Body:** `{"id": "180"}`
	**Note:** Such "id" doesn't exist
5. **Method:** `GET`  or `DELETE`
	**URL:** `http://0.0.0.0:8000/api/actor` or `http://0.0.0.0:8000/api/movie`
	**Body:** `{"id": "one"}`
	**Note:** "id" should be integer
6.  **Method:** `GET` or `DELETE` 
	**URL:** `http://0.0.0.0:8000/api/actor` or `http://0.0.0.0:8000/api/movie`
	**Body:** `{}`
	**Note:** No "id" specified
	
7. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"id": "1", "date_of_birth": "02.20.1970"}`
	**Note:** Wrong date format (20th month specified)
8. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"id": "1", "height": "180"}`
	**Note:** Wrong input field ("height")
9.  **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/movie`
	**Body:** `{id": 1, "length": "120"}`
	**Note:** Wrong input field ("length")

10. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"id": 180, "name": "Megan Fox Fox"}`
	**Note:** Such "id" doesn't exist
11. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"id": "one", "name": "Megan Fox Fox"}` 
	**Note:** "id" should be integer
12. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/actor`
	**Body:** `{"name": "Megan Fox Fox"}`
	**Note:** No "id" specified
	
13. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/actor-relations` or `http://0.0.0.0:8000/api/movie-relations`
	**Body:** `{"id": 180, "relation_id": 1}` or `{"id": 1, "relation_id": 180}`
	**Note:** Such "id" doesn't exist
14. **Method:** `PUT` 
	**URL:** `http://0.0.0.0:8000/api/actor-relations` or `http://0.0.0.0:8000/api/movie-relations`
	**Body:** `{"id": "one", "relation_id": 1}` or `{"id": 1, "relation_id": "one"}`
	**Note:** "id" should be integer
15. **Method:** `PUT` 
	**URL:**`http://0.0.0.0:8000/api/actor-relations` or `http://0.0.0.0:8000/api/movie-relations`
	**Body:** `{"id": 1}` or `{"relation_id": 1}`
	**Note:** No "id" specified

13. **Method:** `DELETE` 
	**URL:** `http://0.0.0.0:8000/api/actor-relations` or `http://0.0.0.0:8000/api/movie-relations`
	**Body:** `{"id": 180}` 
	**Note:** Such "id" doesn't exist
14. **Method:** `DELETE` 
	**URL:** `http://0.0.0.0:8000/api/actor-relations` or `http://0.0.0.0:8000/api/movie-relations`
	**Body:** `{"id": "one"}`
	**Note:** "id" should be integer
15. **Method:** `DELETE`  
	**URL:**`http://0.0.0.0:8000/api/actor-relations` or `http://0.0.0.0:8000/api/movie-relations`
	**Body:** `{}`
	**Note:** No "id" specified


## Dockerize your app

In this section, we'll create our own Docker container for the created application.
So, you'll need the following packages:
```
SQLAlchemy
Flask
Flask_SQLAlchemy
psycopg2
```

Create a folder `app` that will serve as a root for your project. Create `requirements.txt` file and put there names of packages from above  

In the root of your project create a new text file and write the following commands:
```
FROM python:3.7

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

RUN export PYTHONPATH='${PYTHONPATH}:/app'

COPY . .

CMD ["python", "./run.py"]
```
Upon doing this, build the Docker image with the following command in the root of your project:
`docker build -t <user-name>/<name-of-the-container>:<tag-name> .`   
and run it:  
`docker run --network=host --env DB_URL=postgresql+psycopg2://test_user:password@localhost/test_db -p 8000:8000 <user-name>/<name-of-the-container>:<tag-name>`

Now your application is running in the docker container!
You should test it the same way you did it earlier.

To submit the project, push the image to the [**Docker Hub**](https://hub.docker.com/) using:
`docker push <user-name>/<name-of-the-container>:<tag-name>`

Then provide its name to the `@DRU Bot`.

If you have any questions, write `@DRU Team` in Slack!
