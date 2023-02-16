# File: env.py
# Description: Building the environment-2 for the Mobile Robot to explore
# Agent - Mobile Robot
# Obstacles - 'road closed', 'trees', 'traffic lights', 'buildings'
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Reinforcement Learning Algorithms for global path planning // GitHub platform. DOI: 10.5281/zenodo.1317899



# Importing libraries
import numpy as np  # To deal with data in form of matrices
import tkinter as tk  # To build GUI
import time  # Time is needed to slow down the agent and to see how he runs
from PIL import Image, ImageTk  # For adding images into the canvas widget


# Setting the sizes for the environment
pixels = 20   # pixels
env_height = 25  # grid height
env_width = 25  # grid width



# Global variable for dictionary with coordinates for the final route
a = {}


# Creating class for the environment
class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title('RL Q-learning. Sichkar Valentyn')
        self.geometry('{0}x{1}'.format(env_height * pixels, env_height * pixels))
        self.build_environment()

        # Dictionaries to draw the final route
        self.d = {}
        self.f = {}

        # Key for the dictionaries
        self.i = 0

        # Writing the final dictionary first time
        self.c = True

        # Showing the steps for longest found route
        self.longest = 0

        # Showing the steps for the shortest route
        self.shortest = 0

    # Function to build the environment
    def build_environment(self):
        self.canvas_widget = tk.Canvas(self,  bg='white',
                                       height=env_height * pixels,
                                       width=env_width * pixels)

        # Uploading an image for background
        img_background = Image.open("images/bg.png")
        self.background = ImageTk.PhotoImage(img_background)
        # Creating background on the widget
        self.bg = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.background)

        # Creating grid lines
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')

        # Creating objects of  Switches
        # An array to help with building rectangles
        self.o = np.array([pixels / 2, pixels / 2])
        self.r1 = (np.random.randint(1, 22, size=3)) * 20
        self.r2 = (np.random.randint(1, 22, size=3)) * 20
        self.n = self.o + np.array([self.r1[2], self.r2[2]])

        # Obstacle 1
        # Defining the center of obstacle 1
        obstacle1_center = self.o + np.array([self.r1[0],self.r2[0]])
        # Building the obstacle 1
        self.obstacle1 = self.canvas_widget.create_rectangle(
            obstacle1_center[0] - 10, obstacle1_center[1] - 10,  # Top left corner       ****크기****
            obstacle1_center[0] + 10, obstacle1_center[1] + 10,  # Bottom right corner
            outline='grey', fill='#00BFFF')
        # Saving the coordinates of obstacle 1 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle1 = [self.canvas_widget.coords(self.obstacle1)[0] + 10,
                                 self.canvas_widget.coords(self.obstacle1)[1] + 10,
                                 self.canvas_widget.coords(self.obstacle1)[2] - 10,
                                 self.canvas_widget.coords(self.obstacle1)[3] - 10]

        # Obstacle 2
        # Defining the center of obstacle 1

        obstacle2_center = self.o + np.array([self.r1[1], self.r2[1]])
        # Building the obstacle 1
        self.obstacle2 = self.canvas_widget.create_rectangle(
            obstacle2_center[0] - 10, obstacle2_center[1] - 10,  # Top left corner       ****크기****
            obstacle2_center[0] + 10, obstacle2_center[1] + 10,  # Bottom right corner
            outline='grey', fill='#3DBA4A')
        # Saving the coordinates of obstacle 1 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle2 = [self.canvas_widget.coords(self.obstacle2)[0] + 10,
                                 self.canvas_widget.coords(self.obstacle2)[1] + 10,
                                 self.canvas_widget.coords(self.obstacle2)[2] - 10,
                                 self.canvas_widget.coords(self.obstacle2)[3] - 10]
        obstacle2_center = self.o + np.array([self.r1[1], self.r2[1]])



        # Creating an agent of Mobile Robot - red point

        self.agent = self.canvas_widget.create_oval(
            self.n[0] - 7, self.n[1] - 7,
            self.n[0] + 7, self.n[1] + 7,
            outline='#FF1493', fill='#FF1493')


        # Final Point - yellow point
        flag_center = self.o + np.array([self.r1[2], self.r2[2]])
        # Building the flag
        self.flag = self.canvas_widget.create_rectangle(
            flag_center[0] - 10, flag_center[1] - 10,  # Top left corner
            flag_center[0] + 10, flag_center[1] + 10,  # Bottom right corner
            outline='grey', fill='yellow')
        # Saving the coordinates of the final point according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_flag = [self.canvas_widget.coords(self.flag)[0] + 3,
                            self.canvas_widget.coords(self.flag)[1] + 3,
                            self.canvas_widget.coords(self.flag)[2] - 3,
                            self.canvas_widget.coords(self.flag)[3] - 3]

        # Packing everything
        self.canvas_widget.pack()

    # Function to reset the environment and start new Episode
    def reset(self):
        self.update()
        #time.sleep(0.5)

        # Updating agent
        self.canvas_widget.delete(self.agent)
        self.agent = self.canvas_widget.create_oval(
            self.n[0] - 7, self.n[1] - 7,
            self.n[0] + 7, self.n[1] + 7,
            outline='red', fill='red')

        # Clearing the dictionary and the i
        self.d = {}
        self.i = 0

        # Return observation
        return self.canvas_widget.coords(self.agent)

    # Function to get the next observation and reward by doing next step
    def step(self, action):
        # Current state of the agent
        state = self.canvas_widget.coords(self.agent)
        base_action = np.array([0, 0])

        # Updating next state according to the action
        # Action 'up'
        if action == 0:
            if state[1] >= pixels:
                base_action[1] -= pixels
        # Action 'down'
        elif action == 1:
            if state[1] < (env_height - 1) * pixels:
                base_action[1] += pixels
        # Action right
        elif action == 2:
            if state[0] < (env_width - 1) * pixels:
                base_action[0] += pixels
        # Action left
        elif action == 3:
            if state[0] >= pixels:
                base_action[0] -= pixels



        # Moving the agent according to the action
        self.canvas_widget.move(self.agent, base_action[0], base_action[1])

        # Writing in the dictionary coordinates of found route
        self.d[self.i] = self.canvas_widget.coords(self.agent)

        # Updating next state
        next_state = self.d[self.i]

        # Updating key for the dictionary
        self.i += 1

        # Calculating the reward for the agent
        if next_state == self.coords_flag:
            time.sleep(0.1)
            reward = 1
            done = True
            next_state = 'goal'

            # Filling the dictionary first time
            if self.c == True:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)

            # Checking if the currently found route is shorter
            if len(self.d) < len(self.f):
                # Saving the number of steps for the shortest route
                self.shortest = len(self.d)
                # Clearing the dictionary for the final route
                self.f = {}
                # Reassigning the dictionary
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]

            # Saving the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)

        elif next_state in [self.coords_obstacle1,
                            self.coords_obstacle2]:

            reward = -1
            done = True
            next_state = 'obstacle'

            # Clearing the dictionary and the i
            self.d = {}
            self.i = 0

        else:
            reward = 0
            done = False

        return next_state, reward, done

    # Function to refresh the environment
    def render(self):
        #time.sleep(0.03)
        self.update()

    # Function to show the found route
    def final(self):
        # Deleting the agent at the end
        self.canvas_widget.delete(self.agent)

        # Showing the number of steps
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)

        # Creating initial point
        self.initial_point = self.canvas_widget.create_oval(
            self.n[0] - 4, self.n[1] - 4,
            self.n[0] + 4, self.n[1] + 4,
            fill='blue', outline='blue')

        # Filling the route
        for j in range(len(self.f)):
            # Showing the coordinates of the final route
            print(self.f[j])
            self.track = self.canvas_widget.create_oval(
                self.f[j][0] - 3 + self.n[0] - 4, self.f[j][1] - 3 + self.n[1] - 4,
                self.f[j][0] - 3 + self.n[0] + 4, self.f[j][1] - 3 + self.n[1] + 4,
                fill='blue', outline='blue')
            # Writing the final route in the global variable a
            a[j] = self.f[j]


# Returning the final dictionary with route coordinates
# Then it will be used in agent_brain.py
def final_states():
    return a


# This we need to debug the environment
# If we want to run and see the environment without running full algorithm
if __name__ == '__main__':
    env = Environment()
    env.mainloop()
