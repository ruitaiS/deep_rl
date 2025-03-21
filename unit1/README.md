## Lunar Lander RL Model

Adapted largely from:
https://huggingface.co/learn/deep-rl-course/en/unit1/hands-on

### Observation Space:
0) Horizontal pad coordinate (x)
1) Vertical pad coordinate (y)
2) Horizontal speed (x)
3) Vertical speed (y)
4) Angle
5) Angular speed
6) If the left leg contact point has touched the land (boolean)
7) If the right leg contact point has touched the land (boolean)

### Action Space:
0) Do nothing,
1) Fire left orientation engine,
2) Fire the main engine,
3) Fire right orientation engine.

### Reward function:

After every step a reward is granted. The total reward of an episode is the sum of the rewards for all the steps withinthat episode.

For each step, the reward:

- Is increased/decreased the closer/further the lander is to the landing pad.
- Is increased/decreased the slower/faster the lander is moving.
- Is decreased the more the lander is tilted (angle not horizontal).
- Is increased by 10 points for each leg that is in contact with the ground.
- Is decreased by 0.03 points each frame a side engine is firing.
- Is decreased by 0.3 points each frame the main engine is firing.

The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

An episode is considered a solution if it scores at least 200 points. The score is taken as the mean reward score minus the standard deviation



