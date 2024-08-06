import sys

def calculate_state_size(state):
    """
    Function to calculate the size of one state class.

    @param state - The state object.
    @return the size in bytes
    """
    obs_size = (state.observation.numel() * state.observation.element_size())
    return obs_size + \
            sys.getsizeof(state.reward) + \
            sys.getsizeof(state.done) + \
            sys.getsizeof(state.mask)

def replay_control(agent, M_replay=10000, min_replay_size=int(5e5), max_replay_size=int(1e6)):
    """
    Function to control the replay buffer size per episode.

    @param agent - Agent for which the replay buffer needs to be resized.
    @param M_replay - Replay Buffer memory budget.
    @return new calculated size.
    """

    if agent is None:
        return 0.0

    # Equation 3 from the paper
    if len(agent.replay_buffer.buffer) == 0:
        state_size = 1
    else:
        state, action, next_state = agent.replay_buffer.buffer[0]
        state_size = calculate_state_size(state) + \
            calculate_state_size(next_state) + \
            sys.getsizeof(action)

    new_size = int(M_replay / state_size)

    if new_size < min_replay_size:
        new_size = min_replay_size
    if new_size > max_replay_size:
        new_size = max_replay_size

    agent.replay_buffer.buffer_resize(new_size)

    return new_size
