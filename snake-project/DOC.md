# Snake Env Tutorial

*Greetings, my young Padawan! Welcome to the glorious battle in the war for **Data Science Fundamentals**! 
The aim of this project is to show you the basic usage of **Python** and do some fun practice. We tried to make it in the most interesting way, so we hope you will like it :)*

# Table Of Contents
-  [Intro](#intro)
-  [Project Structure](#project-structure)
-  [Main Components](#main-components)
    -  [Gym Interface](#gym-interface)
    -  [Before we start](#before-we-start)
    -  [Constants](#constants)
    -  [Snake Entity](#snake-entity)
    -  [World Entity](#world-entity)
    -  [Renderer](#renderer)
    -  [Environment](#environment)
    -  [Interactor](#interactor)


## Intro

The second module is dedicated to studying **pure Python**, so today we will use a minimum set of additional packages. 
You probably already know the topic of our project, so it's not a surprise for you - of course, it's a **Snake game**! But today it won't be a classic approach because we will implement a **Gym environment** for this game.
The **[Gym](https://gym.openai.com/)** is a toolkit for [**RL**](https://medium.com/ai%C2%B3-theory-practice-business/reinforcement-learning-part-1-a-brief-introduction-a53a849771cf) (Reinforcement Learning) which provides a list of **environments** as well as a **common interface** to create custom ones. These environments are used by the reinforcement learning system as something from which they **gather data** and **learn** how to behave optimally. The toolkit comes with some **pre-built environments** divided into sections: 

- [Algorithms](https://gym.openai.com/envs/#algorithmic): imitate computations
- [Atari](https://gym.openai.com/envs/#atari): Atari 2600 games
- [Box2D](https://gym.openai.com/envs/#box2d): continuous control tasks in the Box2D simulator
- [Classic control](https://gym.openai.com/envs/#classic_control): control theory problems from the classic RL literature
- [MuJoCo](https://gym.openai.com/envs/#mujoco): continuous control tasks, running in a fast physics simulator
- [Robotics](https://gym.openai.com/envs/#robotics): simulated [goal-based tasks](https://blog.openai.com/ingredients-for-robotics-research/) for the Fetch and ShadowHand robots
- [Toy text](https://gym.openai.com/envs/#toy_text): simple text environments to get you started

Every **environment** contains all the **necessary functionality** to **run an agent** and allow it to **learn**. And don't worry, the implementation of an environment is not RL itself. It won't go beyond the concepts you're not familiar with, but in case you'll be working with the RL system, it will give you useful knowledge for future work. In more advanced DRU courses **Gym** and **RL** will be covered in more details :)

**Important note:**
>  The main purpose of the Snake Project is not to show you how to create "Snake"-like games, but to teach you how to use OOP and introduce you to working on complex projects.

Significant part of the functions are already written for you. For sure, the need for some functions will be unclear.
In most cases, bot need them to validate your code correctly. 
So, just keep in mind, there is no necessary to understand it from the beginning to the end.  

## Project Structure

Here we'll define the structure of our project to keep the code organized. 
```

snake
    ├── env                  - contains files with environmental elements.
    │   ├── core             - main parts of env. 
    │   │   ├── snake.py     - snake properties.
    │   │   └── world.py     - grid world properties.
    │   │
    │   ├── utils            - additional parts of env
    │   │   └── renderer.py  - observation rendering tool. 
    │   │
    │   └── snake_env.py     - compilation of main parts of environment.
    │
    ├── settings             - here you can store different constant values, connection parameters, etc.
    │   └── constants.py     - multiple constants storage for their convenient usage.
    │
    ├── local_validator      - validator for your code are stored here
    |   ├── validator        - tests core
    |	|	├── test_constants.py  - some initial and expected constants
    |	|	└── test_validator.py  - main test functions 
    |	|
    |	├── test_snake_step.py - local validator for snake's step method
    |	└── test_world.py      - local validator for world methods
    |    
    └── interactor.py          - script to allow you playing Snake manually.
```

Now define the same structure in your local file system.

## Main Components

### Gym Interface 

Let's consider the [***gym*** interface](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html) to have a clear understanding of what we are going to do:
```python
import gym
from gym import spaces

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2, ...):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    ...
    return observation, reward, done, info

  def render(self, mode='human'):
    ...

  def close (self):
    ...
```

In the **constructor** of the environment class, we need to define the **properties** of an `action_space` like its type and space - it contains all possible **actions** which the **agent** can take in the environment. In our case actions are `left`, `right`, `up`, and `down`. The next required parameter is `observation_space`, which stores all of the **environmental data** to be **observed** by the agent. 
Next is the `step` method which is for **executing** provided **action**, **calculate reward** and **return** resulting **observation**.
And the last is the `render` method which is used to **render** the environment **state**. The `close` method is for closing the rendering.

To make our code **simple** to read we'll hide the **backbone** of main methods in **separate files**. You'll be given the code **templates**, so your task is to **fill** in the **gaps**. But firstly let's define some **constants** we will use in the project.

>**If it still stays unclear - don't worry**
>We will explain everything in detail further in the guide

**Important Notes:**

1. In order to complete this project, it's better for you to have installed [**PyCharm**](https://www.jetbrains.com/pycharm/download/). To install different packages use [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
2. If you are student of any university, you can apply for [**JetBrains Free Educational Licenses**](https://www.jetbrains.com/community/education/#students) and get **PyCharm Professional** for free (only for the period of study)
3. (At the time of writing) if you have **Python 3.9**, when installing ***gym*** you may have version compatibility issues. To fix this you should download **Python 3.8** or older version
---
### Before we start
To understand what we want to have in the end watch this little demo:

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/snake-project/figures/snake.gif?raw=true">
</div>

**As you can see, in the game you have to:**
- Initialize the Snake
- Move Snake with some increasing speed in the direction of its head
- Control the snake with buttons 
- Spawn food randomly 
- Grow Snake after eating food
- Defeat in case of eating itself or the wall
---
### Constants

There are some unchanging things in the world of Snake:

- shape of the world
- initial size of the snake
- objects types
- movement directions
- actions rewards

Move to the **root** of your project and open `settings/constants.py` and paste this code into the file:
```python
import numpy as np

SIZE = (32, 32) # size of the grid world
SNAKE_SIZE = 3 # initial size of the snake
# numeric representation for different types of objects 
# later you will see, that this numbers define a block colour
WALL = 255 # white
FOOD_BLOCK = 64 # red
SNAKE_BLOCK = 100 # green
"""
DIRECTIONS:
    0: UP
    1: RIGHT
    2: DOWN
    3: LEFT
"""
# remember this DIRECTIONS values, we'll use it soon 
DIRECTIONS = [np.array([-1, 0]),
              np.array([0, 1]),
              np.array([1, 0]),
              np.array([0, -1])]
# rewards for different events
DEAD_REWARD = -1.0
MOVE_REWARD = 0.0
EAT_REWARD = 1.0
```

In this project, we will use [`numpy`](https://numpy.org/) package for convenient indexing of the Snake's grid world matrix. You will work with it more in the next module. For now, it will be enough if you go through this [**documentation**](https://numpy.org/devdocs/user/basics.indexing.html).

---
### Snake Entity
Let's initialize the Snake we need to set a bunch of parameters that the environment will give to our snake:
- head position
- current direction
- length

**Important Note:**
>Don't modify given methods and classes, except cases, when it's specified!


In order to do this, we need to open the `env/core/snake.py`:
```python
import numpy as np

from settings.constants import DIRECTIONS, SNAKE_BLOCK


class Snake:
    def __init__(self, head_position, direction_index, length):
        """
        @param head_position: tuple
        @param direction_index: int
        @param length: int
        """
        # Information snake need to know to make the move
        self.snake_block = SNAKE_BLOCK
        self.current_direction_index = direction_index
        # Alive identifier
        self.alive = True
        # Place the snake
        self.blocks = [head_position]
        current_position = np.array(head_position)
        for i in range(1, length):
            # Direction inverse of moving
            current_position = current_position - DIRECTIONS[self.current_direction_index]
            self.blocks.append(tuple(current_position))
            
    def step(self, action):
        # Execute one-time step within the environment
                ...
```
**Clarifying what's going on here:**
1. The snake itself is nothing but array of tuples of its coordinates . This array is stored in `self.blocks` 
2. Initializing `Snake` we added its head first
3. Then went through the all `Snake` length adding other blocks, that lie in the opposite direction to the direction of its head
4. `DIRECTIONS` is an array, that contains `np.arrays` of "changes" of coordinates while moving in the certain direction
5.  `direction_index` is *int* in range from 0 to 3, that defines the direction of Snake's moving. More precisely, it defines the index in `DIRECTIONS` of "change" of coordinates, while moving in certain direction. 

>**Important Note:**
>Further along the course, not everything will be explained in as much detail as here.
So, you need to get used to understand complex things that don't have a clear description
---
**Now we need to implement `step` method**
In `interactor.py` you'll see, that snake moves one block in the direction of it's head every (by default) 0.2 sec
This Snake's single movement is implemented in `step` method
Parameter `action` is the direction index of its head, which is determined by pressing a certain button on the keyboard:

- `⬆` is for up - direction_index is `0`
- `⬇` is for down - direction_index is `2`
- `⮕` if for right - direction_index is `1`
- `⬅` is for left - direction_index is `3`


Use the following template:
```python
import numpy as np

from settings.constants import DIRECTIONS, SNAKE_BLOCK


class Snake:
    def __init__(self, head_position, direction_index, length):
        """
        @param head_position: tuple
        @param direction_index: int
        @param length: int
        """
        # Information snake need to know to make the move
        self.snake_block = SNAKE_BLOCK
        self.current_direction_index = direction_index
        # Alive identifier
        self.alive = True
        # Place the snake
        self.blocks = [head_position]
        current_position = np.array(head_position)
        for i in range(1, length):
            # Direction inverse of moving
            current_position = current_position - DIRECTIONS[self.current_direction_index]
            self.blocks.append(tuple(current_position))

    def step(self, action):
        """
        @param action: int 
        @param return: tuple, tuple
        """
        # Check if action can be performed (do nothing if in the same direction or opposite)
        # Example: if snake looks left, pressing "left" or "right" buttons should change nothing
        if () and ():
            self.current_direction_index = action
        # Remove tail (can be implemented in 1 line)
        tail = 
        self.blocks = 
        # Create new head
        new_head = 
        # Add new head
        # Note: all Snake's coordinates should be tuples (X, Y)
        self.blocks = [new_head] + self.blocks
        return new_head, tail
```
**Let's test it right now!**

To avoid misunderstandings and write code correctly in the future, you should play around with your code and test it in different ways. 
Download [local_validator](https://dru-bot.s3.eu-central-1.amazonaws.com/local_validator.zip) and put it into the root of your project ([Project Structure](#project-structure))

Then run `local_validator/test_snake_step.py`
If your Snake moves correctly and have correct types it will print your Snake movements as a response to commands
>**Expected output:**

```
Lets try to use your step method:

Your Snake initial position: [(5, 5), (5, 4), (5, 3)]
And Snake initial direction: 1

Pressing RIGHT button
Now your Snake position: [(5, 6), (5, 5), (5, 4)]
And Snake direction: 1
Direction didnt change

Pressing LEFT button
Now your Snake position: [(5, 7), (5, 6), (5, 5)]
And Snake direction: 1
Direction didnt change 

Pressing DOWN button
Now your Snake position: [(6, 7), (5, 7), (5, 6)]
And Snake direction: 2
Direction changed

Pressing UP button
Now your Snake position: [(7, 7), (6, 7), (5, 7)]
And Snake direction: 2
Direction didnt change

All actions were completed correctly
Well done!
```

If your code is incorrect, validator will explain your mistake 
>**Example:**
```
Wrong type of coordinates:
They all should be tuples
But you have: <class 'list'>
```
Anyway, if explanations are unclear of just if you are curious, you can figure out how it works examining validator's code. There're no magic :)

>**Note:**
> If local validator disapproves your code, Bot will do as well!

Great! Let's move further and implement the World properties.

---
### World Entity
One important intuition you should have about "Snake" game, that the World of Snake is just a matrix of blocks (Snake blocks, Wall blocks, Food blocks and World blocks)

In the `World` module we need to define such methods as:

- `init_snake`  - creating Snake object
- `init_food` - randomly spawning food
- `get_observation` - get World's copy with placed Snake
- `move_snake` - executing Snake's single movement

`env/core/world.py`:
```python
import numpy as np
import random

from settings.constants import DIRECTIONS, SNAKE_SIZE, DEAD_REWARD, \
    MOVE_REWARD, EAT_REWARD, FOOD_BLOCK, WALL
from env.core.snake import Snake


class World(object):
    def __init__(self, size, custom, start_position, start_direction_index, food_position):
            ...

    def init_snake(self):
            ...

    def init_food(self):
	        ...

    def get_observation(self):
	        ...

    def move_snake(self, action):
	        ...
```

The `snake` and the `food` are usually randomly initialized. But in order to check your project, we need to be able to set locations of the objects manually. So we'll add an identifier to check initialization.

Let's implement the constructor:
```python
import numpy as np
import random

from settings.constants import DIRECTIONS, SNAKE_SIZE, DEAD_REWARD, \
    MOVE_REWARD, EAT_REWARD, FOOD_BLOCK, WALL
from env.core.snake import Snake


class World(object):
    def __init__(self, size, custom, start_position, start_direction_index, food_position):
        """
        @param size: tuple
        @param custom: bool
        @param start_position: tuple
        @param start_direction_index: int
        @param food_position: tuple
        """
        # for custom init
        self.custom = custom
        self.start_position = start_position
        self.start_direction_index = start_direction_index
        self.food_position = food_position
        self.current_available_food_positions = None
        # rewards
        self.DEAD_REWARD = DEAD_REWARD
        self.MOVE_REWARD = MOVE_REWARD
        self.EAT_REWARD = EAT_REWARD
        self.FOOD = FOOD_BLOCK
        self.WALL = WALL
        self.DIRECTIONS = DIRECTIONS
        # Init a numpy ndarray with zeros of predefined size - that will be the initial World
        self.size = size
        self.world = 
        # Fill in the indexes gaps to add walls along the World's boundaries
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        # Get available positions for placing food  
		# Food should not to be spawned in the Walls
        self.available_food_positions = set(zip(*np.where(self.world == 0)))
        # Init snake
        self.snake = self.init_snake()
        # Set food
        self.init_food() 
  
    def init_snake(self):
            ...

    def init_food(self):
        ...


    def get_observation(self):
        ...

    def move_snake(self, action):
        ...
```

Now let's move to the snake and food initialization:
```python
import numpy as np
import random

from settings.constants import DIRECTIONS, SNAKE_SIZE, DEAD_REWARD, \
    MOVE_REWARD, EAT_REWARD, FOOD_BLOCK, WALL
from env.core.snake import Snake


class World(object):
    def __init__(self, size, custom, start_position, start_direction_index, food_position):
        """
        @param size: tuple
        @param custom: bool
        @param start_position: tuple
        @param start_direction_index: int
        @param food_position: tuple
        """
        # for custom init
        self.custom = custom
        self.start_position = start_position
        self.start_direction_index = start_direction_index
        self.food_position = food_position
        self.current_available_food_positions = None
        # rewards
        self.DEAD_REWARD = DEAD_REWARD
        self.MOVE_REWARD = MOVE_REWARD
        self.EAT_REWARD = EAT_REWARD
        self.FOOD = FOOD_BLOCK
        self.WALL = WALL
        self.DIRECTIONS = DIRECTIONS
        # Init a numpy ndarray with zeros of predefined size - that will be the initial World
        self.size = size
        self.world = 
        # Fill in the indexes gaps to add walls along the World's boundaries
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        # Get available positions for placing food  
		# Food should not to be spawned in the Walls
        self.available_food_positions = set(zip(*np.where(self.world == 0)))
        # Init snake
        self.snake = self.init_snake()
        # Set food
        self.init_food()

    def init_snake(self):
        """
        Initialize a snake
        """         
        if not self.custom:
        	# Choose a random position for spawn the Snake
        	# Tail should not spawn outside of the box or in the wall   
			# Remember, coordinates is a tuple(X, Y)
            start_position = 
            # Choose a random direction index
            start_direction_index = 
            new_snake = Snake(start_position, start_direction_index, SNAKE_SIZE)
        else:
            new_snake = Snake(self.start_position, self.start_direction_index, SNAKE_SIZE)
        return new_snake

    def init_food(self):
        """
        Initialize a piece of food
        """
        snake = self.snake if self.snake.alive else None
        # Update available positions for food placement considering snake location 
        # Food should not be spawned in the Snake
        # self.current_available_food_positions should be the set
        self.current_available_food_positions= 
        if not self.custom:
            # Choose a random position from available now
            chosen_position = 
        else:
            chosen_position = self.food_position
            # Code needed for checking your project. Just leave it as it is
            try:
                self.current_available_food_positions.remove(chosen_position)
            except:
                if (self.food_position[0] - 1, self.food_position[1]) in self.current_available_food_positions:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1])
                else:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1] + 1)
                self.current_available_food_positions.remove(chosen_position)
        self.world[chosen_position[0], chosen_position[1]] = self.FOOD
        self.food_position = chosen_position


    def get_observation(self):
        ...

    def move_snake(self, action):
        ...
```


Great! It's time to `get_observation` method. We just need to go through the snake blocks positions array and replace related indexes with the `SNAKE_BLOCK` value:
```python
import numpy as np
import random

from settings.constants import DIRECTIONS, SNAKE_SIZE, DEAD_REWARD, \
    MOVE_REWARD, EAT_REWARD, FOOD_BLOCK, WALL
from env.core.snake import Snake


class World(object):
    def __init__(self, size, custom, start_position, start_direction_index, food_position):
        """
        @param size: tuple
        @param custom: bool
        @param start_position: tuple
        @param start_direction_index: int
        @param food_position: tuple
        """
        # for custom init
        self.custom = custom
        self.start_position = start_position
        self.start_direction_index = start_direction_index
        self.food_position = food_position
        self.current_available_food_positions = None
        # rewards
        self.DEAD_REWARD = DEAD_REWARD
        self.MOVE_REWARD = MOVE_REWARD
        self.EAT_REWARD = EAT_REWARD
        self.FOOD = FOOD_BLOCK
        self.WALL = WALL
        self.DIRECTIONS = DIRECTIONS
        # Init a numpy ndarray with zeros of predefined size - that will be the initial World
        self.size = size
        self.world = 
        # Fill in the indexes gaps to add walls along the World's boundaries
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        # Get available positions for placing food  
		# Food should not to be spawned in the Walls
        self.available_food_positions = set(zip(*np.where(self.world == 0)))
        # Init snake
        self.snake = self.init_snake()
        # Set food
        self.init_food()

    def init_snake(self):
        """
        Initialize a snake
        """         
        if not self.custom:
        	# Choose a random position for spawn the Snake
        	# Tail should not spawn outside of the box or in the wall   
			# Remember, coordinates is a tuple(X, Y)
            start_position = 
            # Choose a random direction index
            start_direction_index = 
            new_snake = Snake(start_position, start_direction_index, SNAKE_SIZE)
        else:
            new_snake = Snake(self.start_position, self.start_direction_index, SNAKE_SIZE)
        return new_snake

    def init_food(self):
        """
        Initialize a piece of food
        """
        snake = self.snake if self.snake.alive else None
        # Update available positions for food placement considering snake location 
        # Food should not be spawned in the Snake
        # self.current_available_food_positions should be the set
        self.current_available_food_positions= 
        if not self.custom:
            # Choose a random position from available now
            chosen_position = 
        else:
            chosen_position = self.food_position
            # Code needed for checking your project. Just leave it as it is
            try:
                self.current_available_food_positions.remove(chosen_position)
            except:
                if (self.food_position[0] - 1, self.food_position[1]) in self.current_available_food_positions:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1])
                else:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1] + 1)
                self.current_available_food_positions.remove(chosen_position)
        self.world[chosen_position[0], chosen_position[1]] = self.FOOD
        self.food_position = chosen_position

    def get_observation(self):
        """
        Get observation of current world state
        """
        obs = self.world.copy()
        snake = self.snake if self.snake.alive else None
        # Here we placing Snake on the World grid with SNAKE_BLOCKs
        if snake:
            for block in snake.blocks:
                obs[block[0], block[1]] = snake.snake_block
                        # snakes head
            obs[snake.blocks[0][0], snake.blocks[0][1]] = snake.snake_block + 1
        return obs

    def move_snake(self, action):
        ...
```
`move_snake` is the most responsible part. Here we need to execute action checking all conditions and calculate reward. Implement the method using the template:
```python
import numpy as np
import random

from settings.constants import DIRECTIONS, SNAKE_SIZE, DEAD_REWARD, \
    MOVE_REWARD, EAT_REWARD, FOOD_BLOCK, WALL
from env.core.snake import Snake


class World(object):
    def __init__(self, size, custom, start_position, start_direction_index, food_position):
        """
        @param size: tuple
        @param custom: bool
        @param start_position: tuple
        @param start_direction_index: int
        @param food_position: tuple
        """
        # for custom init
        self.custom = custom
        self.start_position = start_position
        self.start_direction_index = start_direction_index
        self.food_position = food_position
        self.current_available_food_positions = None
        # rewards
        self.DEAD_REWARD = DEAD_REWARD
        self.MOVE_REWARD = MOVE_REWARD
        self.EAT_REWARD = EAT_REWARD
        self.FOOD = FOOD_BLOCK
        self.WALL = WALL
        self.DIRECTIONS = DIRECTIONS
        # Init a numpy ndarray with zeros of predefined size - that will be the initial World
        self.size = size
        self.world = 
        # Fill in the indexes gaps to add walls along the World's boundaries
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        # Get available positions for placing food  
		# Food should not to be spawned in the Walls
        self.available_food_positions = set(zip(*np.where(self.world == 0)))
        # Init snake
        self.snake = self.init_snake()
        # Set food
        self.init_food()

    def init_snake(self):
        """
        Initialize a snake
        """         
        if not self.custom:
        	# Choose a random position for spawn the Snake
        	# Tail should not spawn outside of the box or in the wall   
			# Remember, coordinates is a tuple(X, Y)
            start_position = 
            # Choose a random direction index
            start_direction_index = 
            new_snake = Snake(start_position, start_direction_index, SNAKE_SIZE)
        else:
            new_snake = Snake(self.start_position, self.start_direction_index, SNAKE_SIZE)
        return new_snake

    def init_food(self):
        """
        Initialize a piece of food
        """
        snake = self.snake if self.snake.alive else None
        # Update available positions for food placement considering snake location 
        # Food should not be spawned in the Snake
        # self.current_available_food_positions should be the set
        self.current_available_food_positions= 
        if not self.custom:
            # Choose a random position from available now
            chosen_position = 
        else:
            chosen_position = self.food_position
            # Code needed for checking your project. Just leave it as it is
            try:
                self.current_available_food_positions.remove(chosen_position)
            except:
                if (self.food_position[0] - 1, self.food_position[1]) in self.current_available_food_positions:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1])
                else:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1] + 1)
                self.current_available_food_positions.remove(chosen_position)
        self.world[chosen_position[0], chosen_position[1]] = self.FOOD
        self.food_position = chosen_position

    def get_observation(self):
        """
        Get observation of current world state
        """
        obs = self.world.copy()
        snake = self.snake if self.snake.alive else None
        # Here we placing Snake on the World grid with SNAKE_BLOCKs
        if snake:
            for block in snake.blocks:
                obs[block[0], block[1]] = snake.snake_block
                        # snakes head
            obs[snake.blocks[0][0], snake.blocks[0][1]] = snake.snake_block + 1
        return obs

    def move_snake(self, action):
        """
        Action executing
        """
        # define reward variable
        reward = 0
        # food needed flag
        new_food_needed = False
        # check if snake is alive
        if self.snake.alive:
            # perform a step (from Snake class)
            new_snake_head, old_snake_tail = self.snake.step(action)
            # Check if snake is outside bounds
            if :
                self.snake.alive = False
            # Check if snake eats itself
            elif :
                self.snake.alive = False
            #  Check if snake eats the food
            if :
                # Remove old food
                
                # Add tail again
                # Note: all Snake coordinates should be tuples(X, Y)
                
                # Request to place new food
                new_food_needed = 
                reward = 
            elif self.snake.alive:
                # Didn't eat anything, move reward
                reward = self.MOVE_REWARD
        # Compute done flag and assign dead reward
        done = not self.snake.alive
        reward = reward if self.snake.alive else self.DEAD_REWARD
        # Adding new food
        if new_food_needed:
            self.init_food()
        return reward, done, self.snake.blocks
```

After filling the gaps you should test our methods
Run `local_validator/test_world.py`
If your methods works correctly, it will use them to create a World and Snake, then move it to the food ~~and kill it!~~
>**Expected output:**
```
Checking World initialization...

Your World without Snake looks like:
[[255. 255. 255. 255. 255. 255. 255. 255. 255. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.  64.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255. 255. 255. 255. 255. 255. 255. 255. 255. 255.]]

Checking Snake movements...

Your World with Snake looks like:
[[255. 255. 255. 255. 255. 255. 255. 255. 255. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.  64.   0.   0. 101.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0. 100.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0. 100.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255. 255. 255. 255. 255. 255. 255. 255. 255. 255.]]

After 3 moves LEFT Snake ate the food and moves 1 DOWN
Now your World with Snake looks like: 
[[255. 255. 255. 255. 255. 255. 255. 255. 255. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.  64.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255. 100. 100. 100.   0.   0.   0.   0.   0. 255.]
 [255. 101.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255. 255. 255. 255. 255. 255. 255. 255. 255. 255.]]

As you can see, Snake grew up for a 1 block

And after 1 move LEFT Snake died:
[[255. 255. 255. 255. 255. 255. 255. 255. 255. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.  64.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255.   0.   0.   0.   0.   0.   0.   0.   0. 255.]
 [255. 255. 255. 255. 255. 255. 255. 255. 255. 255.]]

All actions were completed correctly
Well done!
```
If your code works incorrectly, validator will explain your mistake
>**Example:**
```
Snake didnt die eating itself
```
And as always, if explanations are unclear, you can examine validator's test cases by yourself

***Fantastic! We finished with the main elements of the environment! And now below are the rest of the elements with explanatory comments that we will use for the full operation of snake game. You can just copy paste them.***

### Renderer

The `Renderer` translates the world state with block ids into an RGB image and returns an RGB observation or renders the world using the ***gym*** rendering module. Here we'll use some more complex operations with `numpy`, so you need to read the code carefully to understand on the high level what is going on. You will learn `numpy` in details in the next module. 

Open `env/utils/renderer.py` and paste this code into the file:

```python
import numpy as np


class SnakeColor:
    def __init__(self, body_color, head_color):
        self.body_color = body_color
        self.head_color = head_color


class Colored:
    """
    Translate the world state with block ids into an RGB image
    Return an RGB observation or render the world
    """
    def __init__(self, size, zoom_factor):
        # Setting default colors
        self.snake_colors = SnakeColor((0, 204, 0), (0, 77, 0))
        self.zoom_factor = zoom_factor
        self.size = size
        self.height = size[0]
        self.width = size[1]

    def get_color(self, state):
        # Void => BLACK
        if state == 0:
            return 0, 0, 0
        # Wall => WHITE
        elif state == 255:
            return 255, 255, 255
        # Food => RED
        elif state == 64:
            return 255, 0, 0
        else:
            is_head = (state - 100) % 2
            if is_head == 0:
                return self.snake_colors.body_color
            else:
                return self.snake_colors.head_color

    def get_image(self, state):
        # Transform to RGB image with 3 channels
        color_lu = np.vectorize(lambda x: self.get_color(x), otypes=[np.uint8, np.uint8, np.uint8])
        _img = np.array(color_lu(state))
        # Zoom every channel
        _img_zoomed = np.zeros((3, self.height * self.zoom_factor, self.width * self.zoom_factor), dtype=np.uint8)
        for c in range(3):
            for i in range(_img.shape[1]):
                for j in range(_img.shape[2]):
                    _img_zoomed[c, i * self.zoom_factor:i * self.zoom_factor + self.zoom_factor,
                    j * self.zoom_factor:j * self.zoom_factor + self.zoom_factor] = np.full(
                        (self.zoom_factor, self.zoom_factor), _img[c, i, j])
        #  Transpose to get channels as last
        _img_zoomed = np.transpose(_img_zoomed, [1, 2, 0])
        return _img_zoomed


class Renderer:
    """
    Handles the renderer for the environment
    Receive a map from gridworld and transform it into a visible image (applies colors and zoom)
    """
    def __init__(self, size, zoom_factor):
        self.rgb = Colored(size, zoom_factor)
        self.viewer = None

    def render(self, state, close, mode='human'):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.rgb.get_image(state)
        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
```
### Environment

Now we will put all the modules into the ***gym*** interface template. Here is a template with a completed constructor and defined methods. Open `env/snake_env.py` and paste this code into the file:

```python
import gym
from gym import spaces
import numpy as np

from .core.world import World
from .utils.renderer import Renderer
from settings.constants import SIZE


'''
    Configurable single snake environment.
    Parameters:
        - SIZE: size of the world
        - OBSERVATION_MODE: return a raw observation (block ids) or RGB observation
        - OBS_ZOOM: zoom the observation (only for RGB mode, FIXME)
'''


class SnakeEnv(gym.Env):
    metadata = {
        'render': ['human', 'rgb_array'],
        'observation.types': ['raw', 'rgb']
    }

    def __init__(self, size=SIZE, render_zoom=20, custom=False, start_position=None, start_direction_index=None,
                 food_position=None):
        # for custom init
        self.custom = custom
        self.start_position = start_position
        self.start_direction_index = start_direction_index
        self.food_position = food_position
        #  Set size of the game world
        self.SIZE = size
        # Create world
        self.world = World(self.SIZE, self.custom, self.start_position, self.start_direction_index, self.food_position)
        # Init current step for future usage
        self.current_step = 0
        # Init alive flag
        self.alive = True
                # Observation space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.SIZE[0], self.SIZE[1]),
                                            dtype=np.uint8)
        # Action space
        self.action_space = spaces.Discrete(len(self.world.DIRECTIONS))
        #  Set renderer
        self.RENDER_ZOOM = render_zoom
        self.renderer = None

    def step(self, action):
        """
        Execute action
        @param action: int
        @return: np.array (observation after the action), int (reward), bool ('done' flag), np.array (snake)
        """
        # Perform the action
        reward, done, snake = self.world.move_snake(action)
         
        return self.world.get_observation(), reward, done, snake

    def render(self, mode='human', close=False):
        """
        Render environment depending on the mode
        @param mode: str
        @param close: bool
        @return: np.array
        """
        if not close:
            # Renderer lazy loading
            if self.renderer is None:
                self.renderer = Renderer(size=self.SIZE, zoom_factor=self.RENDER_ZOOM)
            return self.renderer.render(self.world.get_observation(), mode=mode, close=False)

    def close(self):
        """
        Close rendering
        """
        if self.renderer:
            self.renderer.close()
            self.renderer = None
```

Upon this we need some interactor to be able to play game manually.

### Interactor

Since our gameplay is very simple, we can define one key for each action, here we suggest to use arrows to control snake's movement direction:

- `⬆` is for up - direction_index is `0`
- `⬇` is for down - direction_index is `2`
- `⮕` if for right - direction_index is `1`
- `⬅` is for left - direction_index is `3`

By the way, you can define any other keys suitable for you.
Here is the code of the interactor `interactor.py`. Just copy this code below into the file:

```python
import random, time

from pyglet.window.key import MOTION_UP, MOTION_DOWN, MOTION_LEFT, MOTION_RIGHT
from env.snake_env import SnakeEnv


def interact():
    """
    Human interaction with the environment
    """
    env = SnakeEnv()
    done = False
    r = 0
    action = random.randrange(4)
    delay_time = 0.2

    # After the first run of the method env.render()
    # env.renderer.viewer obtains an attribute 'window'
    # which is a pyglet.window.Window object
    env.render(mode='human')

    # Use the arrows to control the snake's movement direction
    @env.renderer.viewer.window.event
    def on_text_motion(motion):
        """
        Events to actions mapping
        """
        nonlocal action
        if motion == MOTION_UP:
            action = 0
        elif motion == MOTION_DOWN:
            action = 2
        elif motion == MOTION_LEFT:
            action = 3
        elif motion == MOTION_RIGHT:
            action = 1

    while not done:
        time.sleep(delay_time)
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
        if reward:
            r += reward
            # Speeding up snake after eating food
            delay_time -= 1/6 * delay_time

    return r


if __name__ == '__main__':
    interact()

```

Now you can run the script with the command `python interactor.py` (debug in case of exceptions). 

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/snake-project/figures/snake.gif?raw=true">
</div>

That what you'll see if everything is right. **The terminal window must be active while playing!**

To submit your project to the bot you need to compress your project to `.zip` with the following structure (validator will not accept archive with a different structure):
```
snake.zip
    ├── env                  
    │   ├── core           
    │   │   ├── snake.py    
    │   │   └── world.py   
    │   │
    │   ├── utils            
    │   │   └── renderer.py  
    │   │
    │   └── snake_env.py     
    │
    ├── settings             
    │   └── constants.py 
    │
    ├── local_validator			 		
    |   ├── validator   	 		
    |	|	├── test_constants.py	
    |	|	└── test_validator.py	
    |	|
    |	├── test_snake_step.py		
    |	└── test_world.py
    |    
    └── interactor.py        
```


Upload it to your Google Drive and set rights as it mentioned in the starting guide. Use instructions from the `DRU-bot` and submit the command - then you'll receive results.

If you have any questions, write `@DRU Team` in Slack!
