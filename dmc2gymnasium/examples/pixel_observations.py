from dmc2gymnasium import DMCGym
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper

if __name__ == "__main__":

    # Create gym
    env = DMCGym("cartpole", "swingup")

    # Add pixel observations wrapper
    env = PixelObservationWrapper(
        env, pixels_only=False, render_kwargs={"pixels": {"height": 64, "width": 64}}
    )

    # Do an example episode
    obs = env.reset()
    while True:
        a = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(a)
        state = obs["state"]
        img = obs["pixels"]
        if term or trunc:
            break
