import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time


class DribbleEnv(gym.Env):
    """
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 120
    }

    def __init__(self):
        self.debug = True
        if self.debug:
            logger.set_level(logger.DEBUG)
        self.gravity = -3000
        self.score = 0
        self.window_width = 800
        self.window_height = 600
        self.ball_radius = 50
        self.ball_center_x = 400
        self.ball_center_y = 300
        self.ball_scale = 0.25
        self.impulsive_force = 60000
        self.impulse_duration = 0.02  # 20 ms
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.mass = 1
        self.damping_factor = 0.8
        self.velocity_after_force = 800
        self.number_of_steps = 0
        self.dt = 0.01

        window_low = np.array([0, 0])
        window_high = np.array([800, 600])

        self.action_space = spaces.Box(low=window_low, high=window_high, dtype=np.int)
        self.observation_space = spaces.Dict({
            "ball_center_x": spaces.Discrete(self.window_width),
            "ball_center_y": spaces.Discrete(self.window_height),
            "ball_radius": spaces.Discrete(self.ball_radius)
        })
        self.window_x_bounds = (self.ball_radius, self.window_width - self.ball_radius)
        self.window_y_bounds = (self.ball_radius, self.window_height - self.ball_radius)

        self.seed()
        self.viewer = None
        self.state = (self.ball_center_x, self.ball_center_y, self.ball_radius)
        self.last_step_time = time.time()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_bounding_box(self):
        lower_left = (self.ball_center_x - self.ball_radius, self.ball_center_y - self.ball_radius)
        top_left = (self.ball_center_x - self.ball_radius, self.ball_center_y + self.ball_radius)
        top_right = (self.ball_center_x + self.ball_radius, self.ball_center_y + self.ball_radius)
        lower_right = (self.ball_center_x + self.ball_radius, self.ball_center_y - self.ball_radius)
        return lower_left, top_left, top_right, lower_right

    def is_coord_inside_ball(self, coordinate):
        """
        Using equation of circle: (x-x0)^2 + (y-y0)^2 <= r^2
        """
        return (coordinate[0] - self.ball_center_x) ** 2 + (coordinate[1] - self.ball_center_y) ** 2 <= self.ball_radius ** 2

    def check_wall_collision_and_update_state(self):
        collision_with_ground = self.ball_center_y <= self.window_y_bounds[0]
        collision_with_left_wall = self.ball_center_x <= self.window_x_bounds[0]
        collision_with_right_wall = self.ball_center_x >= self.window_x_bounds[1]

        if collision_with_ground:
            logger.info("collision with ground")
            logger.info("initial velocity: %s", (self.velocity_x, self.velocity_y))
            self.velocity_y = -self.velocity_y * self.damping_factor
            self.ball_center_y = int(self.window_y_bounds[0])
            logger.info("final   velocity: %s", (self.velocity_x, self.velocity_y))

        elif collision_with_left_wall:
            logger.info("collision with left wall")
            self.velocity_x = -self.velocity_x * self.damping_factor
            self.ball_center_x = int(self.window_x_bounds[0])

        elif collision_with_right_wall:
            logger.info("collision with right wall")
            self.velocity_x = -self.velocity_x * self.damping_factor
            self.ball_center_x = int(self.window_x_bounds[1])

    def x_distance_from_center(self, coordinate):
        return abs(self.ball_center_x - coordinate[0])

    def apply_force(self, coordinate):
        """
        impulse = change in momentum i.e. F * dt = m * (vf - vi)

        self.velocity_x = F_x * dt / self.mass + self.velocity_x
        self.velocity_y = (F_y + self.mass * G) * dt / self.mass + self.velocity_y
        """
        reward = -1.0
        if self.is_coord_inside_ball(coordinate=coordinate):
            logger.info("applying force")
            x_distance = self.x_distance_from_center(coordinate=coordinate)
            self.velocity_x = self.velocity_after_force * (x_distance / self.ball_radius)
            self.velocity_y = self.velocity_after_force
            reward = 1.0
        return reward

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        logger.info("\n\n#####action: %s", action)
        logger.info("step: %s", self.number_of_steps)
        logger.info("window bounds: %s", (self.window_x_bounds, self.window_y_bounds))
        logger.info("ball center coordinate: %s", (self.ball_center_x, self.ball_center_y))
        logger.info("velocity: %s", (self.velocity_x, self.velocity_y))
        self.number_of_steps += 1

        self.ball_center_x += int(self.velocity_x * self.dt)
        self.ball_center_y += int(self.velocity_y * self.dt + 1 / 2 * self.gravity * self.dt ** 2)
        self.velocity_y = self.velocity_y + self.gravity * self.dt
        self.check_wall_collision_and_update_state()

        reward = self.apply_force(action)
        self.state = (self.ball_center_x, self.ball_center_y, self.ball_radius)

        return np.array(self.state), reward, False, {}

    def reset(self):
        self.ball_center_x = self.np_random.randint(low=self.window_x_bounds[0], high=self.window_x_bounds[1])
        self.ball_center_y = self.np_random.randint(low=self.window_y_bounds[0], high=self.window_y_bounds[1])
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.state = (self.ball_center_x, self.ball_center_y, self.ball_radius)
        return np.array(self.state)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.window_width, self.window_height)
            self.football = rendering.make_circle(self.ball_radius)

            # add translation
            self.balltrans = rendering.Transform(translation=(self.ball_center_x, self.ball_center_y))
            self.football.add_attr(self.balltrans)

            self.football.set_color(0, 0, 0)
            self.viewer.add_geom(self.football)

        if self.state is None:
            return None

        ball_center_x, ball_center_y, _ = self.state
        self.balltrans.set_translation(ball_center_x, ball_center_y)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
