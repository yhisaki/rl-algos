from rl_algos.buffers import EpisodeBuffer, EpisodicTrainingBatch, ReplayBuffer, TrainingBatch
from rl_algos.experiments import TransitionGenerator
from rl_algos.utils import is_state_terminal
from rl_algos.wrappers import vectorize_env


def example_replay_buffer():
    env = vectorize_env("Swimmer-v3", 1)

    def actor(state):
        return env.action_space.sample()

    interactions = TransitionGenerator(env, actor, max_step=1000)

    buffer = ReplayBuffer(10 ** 4)
    for steps, states, next_states, actions, rewards, dones, info in interactions:
        terminals = is_state_terminal(env, steps, dones)
        for id, (state, next_state, action, reward, terminal, done) in enumerate(
            zip(states, next_states, actions, rewards, terminals, dones)
        ):
            buffer.append(
                id=id,
                state=state,
                next_state=next_state,
                action=action,
                reward=reward,
                terminal=terminal,
                reset=done,
            )
    s = buffer.sample(10)
    batch = TrainingBatch(**s, device="cuda")
    print(batch.reward)
    print(batch[0:5].reward)
    batch[0:5].reward += 100.0
    print(batch.reward)


def example_episode_buffer():
    env = vectorize_env(env_id="Hopper-v3", num_envs=3)

    def actor(state):
        return env.action_space.sample()

    interactions = TransitionGenerator(env, actor, max_step=300)

    buffer = EpisodeBuffer()

    for steps, states, next_states, actions, rewards, dones, info in interactions:
        terminals = is_state_terminal(env, steps, dones)
        for id, (state, next_state, action, reward, terminal, done) in enumerate(
            zip(states, next_states, actions, rewards, terminals, dones)
        ):
            buffer.append(
                id=id,
                state=state,
                next_state=next_state,
                action=action,
                reward=reward,
                terminal=terminal,
                reset=done,
            )

    episodes = buffer.get_episodes()
    batch = EpisodicTrainingBatch(episodes, "cuda")

    print(batch[0].reward)
    batch[0].reward += 100
    print(batch.flatten.reward)
    for episode in batch:
        for transition in reversed(episode):
            print(transition.reset)
        break


if __name__ == "__main__":
    example_replay_buffer()
    example_episode_buffer()
