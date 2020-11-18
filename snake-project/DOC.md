# Snake Env Tutorial

*Greetings, my young Padawan! Welcome to the glorious battle in the war for **Data Science Fundamentals**! 
The aim of this project is to show you the basic usage of **Python** and do some fun practice. We tried to make it in the most interesting way, so we hope you will like it :)*

# Table Of Contents
-  [Intro](#intro)
-  [Project Structure](#project-structure)
-  [Main Components](#main-components)
    -  [Gym Interface](#gym-interface)
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

Every **environment** contains all the **necessary functionality** to **run an agent** and allow it to **learn**. And don't worry, the implementation of an environment is not RL itself. It won't go beyond the concepts you're not familiar with, but in case you'll be working with the RL system, it will give you useful knowledge for future work.

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
    └── interactor.py        - script to allow you playing Snake manually.
```

Now you have to  define the same structure in your local file system.

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

  def reset(self):
    ...
    return observation  # reward, done, info can't be included

  def render(self, mode='human'):
    ...

  def close (self):
    ...
```

In the **constructor** of the environment class, we need to define the **properties** of an `action_space` like its type and space - it contains all possible **actions** which the **agent** can take in the environment. In our case actions are `left`, `right`, `up`, and `down`. The next required parameter is `observation_space`, which stores all of the **environmental data** to be **observed** by the agent. 
The `reset` method is dedicated to **resetting** the environment to the **initial state,** it means we need to reset the steps counter and reward score. Next is the `step` method which is for **executing** provided **action**, **calculate reward** and **return** resulting **observation**.
And the last is the `render` method which is used to **render** the environment **state**. The `close` method is for closing the rendering.

To make our code **simple** to read we'll hide the **backbone** of main methods in **separate files**. You'll be given the code **templates**, so your task is to **fill** in the **gaps**. But firstly let's define some **constants** we will use in the project.

> **Important note**
In order to complete this project, it's better for you to have installed [PyCharm](https://www.jetbrains.com/pycharm/download/). To install different packages use [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

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
WALL = 255
FOOD_BLOCK = 64
SNAKE_BLOCK = 100
"""
DIRECTIONS:
    0: UP
    1: RIGHT
    2: DOWN
    3: LEFT
"""
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

### Snake Entity

Let's start with defining the Snake properties. 

For now, our snake needs to know how to move, so let's implement class `Snake` with method `step`. In order to do this, we need to open the `env/core/snake.py`:
```python
import numpy as np

from settings.constants import DIRECTIONS, SNAKE_BLOCK


class Snake:
    def __init__(self, param1, param2, ...):
        # Information snake need to know to make the move
        ...

    def step(self, action):
        # Execute one-time step within the environment
        ...
```

Now we need to set a bunch of parameters that the environment will give to our snake:

- head position
- direction
- length

Let's finish the constructor:
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

Upon this you need to implement the `step` method using the following template (all **instructions** specified as **comments**):
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
        if () and ():
            self.current_direction_index = action
        # Remove tail
        tail = 
        self.blocks = 
        # Check new head
        new_head = 
        # Add new head
        self.blocks = [new_head] + self.blocks
        return new_head, tail
```

Great! Let's move further and implement the World properties.

### World Entity

In the `World` module we need to define such methods as:

- initialize snake
- initialize food
- get observation
- move snake

`env/core/world.py`:
```python
import numpy as np
import random

from settings.constants import DIRECTIONS, SNAKE_SIZE, DEAD_REWARD, \
    MOVE_REWARD, EAT_REWARD, FOOD_BLOCK, WALL
from .snake import Snake


class World(object):
    def __init__(self, param1, param2, ...):
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
from .snake import Snake


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
        # rewards
        self.DEAD_REWARD = DEAD_REWARD
        self.MOVE_REWARD = MOVE_REWARD
        self.EAT_REWARD = EAT_REWARD
        self.FOOD = FOOD_BLOCK
        self.WALL = WALL
        self.DIRECTIONS = DIRECTIONS
        # Init a numpy matrix with zeros of predefined size
        self.size = size
        self.world = np.zeros(size)
        # Fill in the indexes gaps to add walls to the grid world
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        # Get available positions for placing food (choose all positions where world block = 0)
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
from .snake import Snake


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
        # rewards
        self.DEAD_REWARD = DEAD_REWARD
        self.MOVE_REWARD = MOVE_REWARD
        self.EAT_REWARD = EAT_REWARD
        self.FOOD = FOOD_BLOCK
        self.WALL = WALL
        self.DIRECTIONS = DIRECTIONS
        # Init a numpy matrix with zeros of predefined size
        self.size = size
        self.world = np.zeros(size)
        # Fill in the indexes gaps to add walls to the grid world
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        # Get available positions for placing food (choose all positions where world block = 0)
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
            # choose a random position between [SNAKE_SIZE and SIZE - SNAKE_SIZE]
            start_position = 
            # choose a random direction index
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
        available_food_positions = 
        if not self.custom:
            # Choose a random position from available
            chosen_position = 
        else:
            chosen_position = self.food_position
            # Code needed for checking your project. Just leave it as it is
            try:
                available_food_positions.remove(chosen_position)
            except:
                if (self.food_position[0] - 1, self.food_position[1]) in available_food_positions:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1])
                else:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1] + 1)
                available_food_positions.remove(chosen_position)
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
from .snake import Snake


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
        # rewards
        self.DEAD_REWARD = DEAD_REWARD
        self.MOVE_REWARD = MOVE_REWARD
        self.EAT_REWARD = EAT_REWARD
        self.FOOD = FOOD_BLOCK
        self.WALL = WALL
        self.DIRECTIONS = DIRECTIONS
        # Init a numpy matrix with zeros of predefined size
        self.size = size
        self.world = np.zeros(size)
        # Fill in the indexes gaps to add walls to the grid world
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        # Get available positions for placing food (choose all positions where world block = 0)
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
            # choose a random position between [SNAKE_SIZE and SIZE - SNAKE_SIZE]
            start_position = 
            # choose a random direction index
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
        available_food_positions = 
        if not self.custom:
            # Choose a random position from available
            chosen_position = 
        else:
            chosen_position = self.food_position
            # Code needed for checking your project. Just leave it as it is
            try:
                available_food_positions.remove(chosen_position)
            except:
                if (self.food_position[0] - 1, self.food_position[1]) in available_food_positions:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1])
                else:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1] + 1)
                available_food_positions.remove(chosen_position)
        self.world[chosen_position[0], chosen_position[1]] = self.FOOD
        self.food_position = chosen_position


    def get_observation(self):
        """
        Get observation of current world state
        """
        obs = self.world.copy()
        snake = self.snake if self.snake.alive else None
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
from .snake import Snake


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
        # rewards
        self.DEAD_REWARD = DEAD_REWARD
        self.MOVE_REWARD = MOVE_REWARD
        self.EAT_REWARD = EAT_REWARD
        self.FOOD = FOOD_BLOCK
        self.WALL = WALL
        self.DIRECTIONS = DIRECTIONS
        # Init a numpy matrix with zeros of predefined size
        self.size = size
        self.world = np.zeros(size)
        # Fill in the indexes gaps to add walls to the grid world
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        self.world[] = self.WALL
        # Get available positions for placing food (choose all positions where world block = 0)
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
            # choose a random position between [SNAKE_SIZE and SIZE - SNAKE_SIZE]
            start_position = 
            # choose a random direction index
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
        available_food_positions = 
        if not self.custom:
            # Choose a random position from available
            chosen_position = 
        else:
            chosen_position = self.food_position
            # Code needed for checking your project. Just leave it as it is
            try:
                available_food_positions.remove(chosen_position)
            except:
                if (self.food_position[0] - 1, self.food_position[1]) in available_food_positions:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1])
                else:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1] + 1)
                available_food_positions.remove(chosen_position)
        self.world[chosen_position[0], chosen_position[1]] = self.FOOD
        self.food_position = chosen_position

    def get_observation(self):
        """
        Get observation of current world state
        """
        obs = self.world.copy()
        snake = self.snake if self.snake.alive else None
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

Fantastic! We finished with the main elements of the environment, and now we need to implement the renderer of the environment.

### Renderer

The `Renderer` translates the world state with block ids into an RGB image and returns an RGB observation or renders the world using the ***gym*** rendering module. Here we'll use some more complex operations with `numpy`, so you need to read the code carefully to understand on the high level what is going on. You will learn `numpy` in details in the next module. 

`env/utils/renderer.py`:
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

Now we will put all the modules into the ***gym*** interface template. Here is a template with a completed constructor and defined methods. Your task is to implement methods using instructions:

`env/snake_env.py`
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
        # Check if game is ended
        
        # Perform the action
        reward, done, snake = 
        # Disable interactions if snake dead
        
        return self.world.get_observation(), reward, done, snake

    def reset(self,):
        """
        Reset environment to the initial state
        @return: initial observation
        """
        # Reset step counters

        # Set 'alive' flag
        
        # Create world
        self.world =
        return self.world.get_observation()

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

- `⬆` is for `up` (0)
- `⬇` is for `down` (2)
- `⬅` is for `left` (3)
- `⮕` is for `right` (4)

By the way, you can define any other keys suitable for you.
Here is the code of the interactor `interactor.py`:

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
Archive.zip
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
    └── interactor.py
```

Upload it to your Google Drive and set rights as it mentioned in the starting guide. Use instructions from the `DRU-bot` and submit the command - then you'll receive results.

If you have any questions, write `@DRU Team` in Slack!
