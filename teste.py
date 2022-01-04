import time
import gym
from maze import environment

if __name__ == '__main__':
    env = gym.make('Maze-v0', shape=(10, 10))
    env.reset()
    env.change_start_and_goal() 
    env.render()
    time.sleep(5)
    env.step(0)
    env.render()
    time.sleep(5)