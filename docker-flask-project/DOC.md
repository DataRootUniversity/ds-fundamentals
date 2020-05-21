# RESTful APIs Project Guide

*Hello and welcome to the **RESTful APIs Project** tutorial! During the next 1.5-2 hours you'll learn how to **structure** your project and **develop** your own **web API**.*

There are lots of different ways to structure your RESTful API. Here we have gathered the **best practices** of project organization. During the tutorial, we will provide an example of a well-structured project to show you how to make your code **organized** and **effective**.

# Table Of Contents

- [Intro](#intro)
- [Try it first](#try-it-first)
- [Details](#details)
    - [Folder structure](#folder-structure)
    - [Main Components](#main-components)
        - [Models](#models)
        - [Controllers](#controllers)
        - [Core](#core)
- [Test your app](#test-your-app)
- [Dockerize your app](#dockerize-your-app)

# Intro

In this tutorial, we're going to implement a web **API** (**A**pplication **P**rogramming **I**nterface) using **REST** (**RE**presentational **S**tateless **T**ransfer) approach and based on **MVC** (**M**odel, **V**iew, **C**ontroller) architectural pattern. Our service will allow users to interact with the database and make basic **CRUD** (**C**reate, **R**ead, **U**pdate, **D**elete) operations. We will use **PostgreSQL** as a database because of its robustness and high performance.

This template is based on best structuring patterns, so it can be adapted for any other similar project.

# Try it first

Install all the needed libraries. It's better to use `virtualenv` to keep the general `pip` clean. We are assuming you are using anaconda distribution and a 3.7+ version of python.
So, you'll need the following packages:
```
SQLAlchemy
Flask
Flask_SQLAlchemy
psycopg2
```
> **Hint:**
Create a folder `app` that will serve as a root for your project. Create `requirements.txt` file and put there all packages from bellow - we'll use this file later. Also, you can generate requirements.txt automatically after completing the project.

# Details

As you already know, the main idea of the project is to develop an API for interaction with the database.
So, we'll be able to **add/remove/update** records via API routes.

> **A little spoiler for better understanding of the following material:** 
There will be two entities in the DB: **Actor** and **Movie**, so our main task is to be able to manipulate records and relations between entities through the API.
We'll get familiar with all the details a little bit later.

Now let's dive deeper into the project components structure.

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
    │   ├── __init__.py             - init file for the package.
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
    └── run.py                  - application run file.
```
> Now it's time for you to define the structure by carrying it to your local machine.

## Main components

> In this part, we're going to code a bit. I hope, you've already installed [**PyCharm**](https://www.jetbrains.com/pycharm/download/#section=mac) to your machine.
Also, you'll need to install [**Docker**](https://www.postgresql.org/download/).

### Setting up a database

Firstly we need to set up a database we will work with. In order to implement the project, you need to [install **Postgres**](https://www.postgresql.org/download/) locally to your machine. Also, you need to [create a user and database](https://medium.com/coding-blocks/creating-user-database-and-adding-access-on-postgresql-8bfcd2f4a91e) with the following credentials:
```
DB_USER = 'test_user'
DB_PASS = 'password'
DB_NAME = 'test_db'
```
Upon this you need to create environmental variable `DB_URL` which we will use to connect to the database. Pass this to your command line:
```
export DB_URL=postgresql+psycopg2://test_user:password@0.0.0.0:5432/test_db
```
### Models

When all environmental details are handled, let's clarify entities we will use in the project.
There are two entities: **Actor** and **Movie**. One actor can star in multiple movies, the movie's cast can consist of multiple actors. So the relation is **many-to-many**.
You can see all properties in the schema below:

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/db-diagram.png?raw=true">
</div>


We need to implement **models** of these **entities**.
As you can see, some more constants to appear here, so let's put them all to the `settings/constants.py`:
```python
import os
# connection credentials
DB_URL = os.environ['DB_URL']                                                                     
# entities properties
ACTOR_FIELDS = ['id', 'name', 'gender', 'date_of_birth']
MOVIE_FIELDS = ['id', 'name', 'year', 'genre']

# date of birth format
DATE_FORMAT = '%d.%m.%Y'
```
We'll use [`SQLAlchemy`](https://www.sqlalchemy.org/) toolkit to manipulate all stuff related to the DataBase (defining models, initialization of the DB, etc).

Let's define an **association** table, keeping in mind that our **entities** have **relations** between each other.

> **Important note:**
To start working with the models we need to initialize the `DB` object. Go to the `core/__init__.py` and define it:
```python
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()
```
Now we need to define the relations between entities. Look through the [documentation](https://docs.sqlalchemy.org/en/13/orm/basic_relationships.html') and then move to the `models/relations.py` and implement an **association table** following the instructions below:
```python
from core import db


# Table name -> 'association'
# Columns: 'actor_id' -> db.Integer, db.ForeignKey -> 'actors.id', primary_key = True
#          'movie_id' -> db.Integer, db.ForeignKey -> 'movies.id', primary_key = True
association =
```

Let's try to create our first model `Actor`. Check out [documentation](https://flask-sqlalchemy.palletsprojects.com/en/2.x/models/), then open `models/actor.py` and complete the model's class with defining entities properties:
```python
from datetime import datetime as dt

from core import db
from .relations import association

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
from .relations import association


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
`export PYTHONPATH='${PYTHONPATH}:/path/to/your/project`

In order to check our models, we need to get `db` instance from `core` and initialize some dummy Flask `app`:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime as dt

from settings.constants import DB_URL
from core import db
from models import Actor, Movie


app = Flask(__name__, instance_relative_config=False)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # silence the deprecation warning

db.init_app(app)
```
Now let's add some data to the table. We need to use `app.app_context()` to do this:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime as dt

from settings.constants import DB_URL
from core import db
from models import Actor, Movie


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
    from .base import Model
    from .relations import association
    
    
    class Actor(Model, db.Model):
        . . .
```

and `Movie`:
```python
    from datetime import datetime as dt
    
    from core import db
    from .base import Model
    from .relations import association
    
    
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
from models import Actor, Movie

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
from models import Actor, Movie


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
    print('created actor:', actor.__dict__, '\\n')

    movie = Movie.create(**data_movie)
    print('created movie:', movie.__dict__, '\\n')

    upd_actor = Actor.update(1, **data_actor_upd)
    print('updated actor:', upd_actor.__dict__, '\\n')

    upd_movie = Movie.update(1, **data_movie_upd)
    print('updated movie:', upd_movie.__dict__, '\\n')

    add_rels_actor = Actor.add_relation(1, upd_movie)
    movie_2 = Movie.create(**data_movie)
    add_more_rels_actor = Actor.add_relation(1, movie_2)
    print('relations list:', add_more_rels_actor.filmography, '\\n')

    clear_rels_actor = Actor.clear_relations(1)
    print('all relations cleared:', clear_rels_actor.filmography, '\\n')

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
```python
from flask import request


def get_request_data():
    """
    Get keys & values from request
    """

    return data
```
Now we can move to `controllers/actor.py` and implement operations handlers which will deal with the requests for the `Actor` model.
Don't forget to handle exceptions:

> **Hint:**
Use [`flask.make_response`](https://kite.com/python/docs/flask.make_response) to construct correctly formatted responses. For **correct** response use code `200`, and `400` for **bad** requests.
Here is a **template** for you to implement needed operations handlers (with a few completed examples as a bonus)
```python
from flask import jsonify, make_response

from datetime import datetime as dt
from ast import literal_eval

from models import Actor, Movie
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
Don't forget to **test** your functions! You can just **comment** **out** the line with `get_request_data` call and pass a `dict` to the function like this:
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

    else:
        err = 'No id specified'
        return make_response(jsonify(error=err), 400) 
```
Now it's time to implement `Movie` operations in the same way:
```python
from flask import jsonify, make_response

from ast import literal_eval

from models import Movie, Actor
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
        - `name, gender, date_of_birth`
    - `DELETE`: remove actor, `body`:
        - `id`
- `/api/movie`, to manipulate with the movies records, methods:
    - `GET`: get movie by id
    - `POST`: add new movie, `body` can include:
        - `name, year, genre`
    - `PUT`: update movie, `body` can include:
        - `name, year, genre`
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
    Get all actors in db
    """
    return get_all_actors()

        .  .  .
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
Then you can use some `GUI` for sending requests like [`Postman`](https://www.getpostman.com/downloads/) or `curl` requests in Python.

Here are examples of **correct** requests with `200` response status code:

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/add-actor.png?raw=true">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/add-movie.png?raw=true">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/actors.png?raw=true">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/del-actor.png?raw=true">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/filmography.png?raw=true">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/docker-flask-project/figures/del-filmography.png?raw=true">
</div>


## Dockerize your app

In this section, we'll create our own Docker container for the created application.

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
`docker build -t <name-of-the-container> .`
and run it:
`docker run -p 8000:8000 <name-of-the-container>`

Now your application is running in the docker container!
You can test it the same way you did it earlier.

To submit the project, upload the image to the [`Docker Hub`](https://hub.docker.com/), and provide its name to the `@DRU-bot`.

If you have any questions, DM me (`@Alexandra Severinchik`) in `Slack`!
