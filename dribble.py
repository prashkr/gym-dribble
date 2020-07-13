import gym
import time

env = gym.make('gym_dribble:Dribble-v0')
for i_episode in range(1000):
    observation = env.reset()
    for t in range(1000):
        # time.sleep(0.01)
        env.render()
        # print("observation: ", observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print("reward: ", reward)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
