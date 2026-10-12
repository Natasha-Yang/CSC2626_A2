import numpy

numpy.bool8 = numpy.bool_
import sys
import pickle
import gym
import gym_thing
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import time
import cv2
from dmp import DMP
from policy import BallCatchDMPPolicy

PATH_TO_HOME = str(pathlib.Path(__file__).parent.resolve())
PATH_TO_NLOPTJSON = str(
    (
        pathlib.Path(__file__).parent
        / "ballcatch_env"
        / "gym_thing"
        / "nlopt_optimization"
        / "config"
        / "nlopt_config_stationarybase.json"
    ).resolve()
)

TRAINED_DMP_PATH = "results/trained_dmp.pkl"


def make_env(
    online=True,
    dataset=None,
    n_substeps=1,
    gravity_factor_std=0.0,
    n_substeps_std=0.0,
    mass_std=0.0,
    act_gain_std=0.0,
    joint_noise_std=0.0,
    vicon_noise_std=0.0,
    control_noise_std=0.0,
    random_training_index=False,
):
    # Create thing env
    initial_arm_qpos = np.array([1, -0.3, -1, 0, 1.6, 0])
    initial_ball_qvel = np.array([-1, 1.5, 4.5])
    initial_ball_qpos = np.array([1.22, -1.6, 1.2])
    zero_base_qpos = np.array([0.0, 0.0, 0.0])

    training_data = None
    if not online:
        training_data = np.load(dataset)
        print("OFFLINE")

    env = gym.make(
        "ballcatch-v0",
        model_path="robot_with_cup.xml",
        nlopt_config_path=PATH_TO_NLOPTJSON,
        initial_arm_qpos=initial_arm_qpos,
        initial_ball_qvel=initial_ball_qvel,
        initial_ball_qpos=initial_ball_qpos,
        initial_base_qpos=zero_base_qpos,
        pos_rew_weight=0.1,
        rot_rew_weight=0.01,
        n_substeps=n_substeps,
        online=online,
        training_data=training_data,
        gravity_factor_std=gravity_factor_std,
        n_substeps_std=n_substeps_std,
        mass_std=mass_std,
        act_gain_std=act_gain_std,
        joint_noise_std=joint_noise_std,
        vicon_noise_std=vicon_noise_std,
        control_noise_std=control_noise_std,
        random_training_index=random_training_index,
    )
    env.reset()
    return env


# Camera nudged per key press when rotating/zooming the render window.
CAMERA_KEYS = {
    ord("a"): ("azimuth", -5),
    ord("d"): ("azimuth", 5),
    ord("w"): ("elevation", 5),
    ord("s"): ("elevation", -5),
    ord("z"): ("distance", -0.1),
    ord("x"): ("distance", 0.1),
}


def display_frame(env, window_name="ballcatch", width=480, height=480):
    """Render one offscreen frame (EGL on GPU machines, OSMesa on CPU-only
    ones -- picked automatically, see gym_thing/__init__.py) and show it in
    a normal window.

    Returns False if the user closed the window / pressed q or Esc, so the
    caller can stop trying to render.
    """
    frame = env.render(mode="rgb_array", width=width, height=height)
    cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), 27):  # q or Esc
        cv2.destroyWindow(window_name)
        return False

    if key in CAMERA_KEYS:
        cam = env.unwrapped.cam
        attr, delta = CAMERA_KEYS[key]
        setattr(cam, attr, getattr(cam, attr) + delta)

    return True


def test_policy(eval_env, policy, eval_episodes=5, render_freq=1, seed=1):
    # Set seeds
    eval_env.seed(seed)
    np.random.seed(seed)

    avg_reward = 0.0
    successes = 0
    rendering = render_freq != 0

    for eps in range(eval_episodes):
        print(f"\rEvaluating Episode {eps+1}/{eval_episodes}", end="")
        state, done, truncated = eval_env.reset(), False, False
        policy.set_goal(state=state, goal=eval_env.env.goal)
        while not (done or truncated):
            action = policy.select_action(np.array(state))
            state, reward, done, truncated, info = eval_env.step(action)
            if rendering and eps % render_freq == 0:
                rendering = display_frame(eval_env)
            avg_reward += reward
        if truncated and not done:
            actual_done = False
        else:
            actual_done = done

        if actual_done:
            successes += 1

    cv2.destroyAllWindows()
    avg_reward /= eval_episodes
    success_pct = float(successes) / eval_episodes

    print("")
    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}, success%: {:.3f}".format(eval_episodes, avg_reward, success_pct))
    print("---------------------------------------")
    print("")
    return avg_reward, success_pct


def load_dataset(dataset_path="data/demos.pkl"):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    return [traj[:, :6] for traj in dataset["trajectories"]]  # 1st 6 elements are joint angle


def q2_recons():
    trajectories = load_dataset()
    dt = 0.04
    # TODO: Train a DMP on trajectories[0]
    dmp =
    rollout =

    # for k in range(6):
    #     plt.figure()
    #     plt.plot(demo_time, demo[:, k], label='GT')
    #     plt.plot(rollout_time, rollout[:, k], label='DMP')
    #     plt.legend()
    #     plt.savefig(f'results/recons_{k}.png')



def q2_tuning():
    dt = 0.04
    trajectories = load_dataset()

    # TODO: select the best settings for fitting the demos
    dmp =
    X, T =

    dmp.learn(X, T)
    dmp.save(TRAINED_DMP_PATH)


def main():
    env = make_env(n_substeps=1)
    dmp = DMP.load(TRAINED_DMP_PATH)

    policy = BallCatchDMPPolicy(dmp, dt=env.dt)
    test_policy(env, policy, eval_episodes=20, render_freq=1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "recons":
        q2_recons()
    elif sys.argv[1] == "tuning":
        q2_tuning()
