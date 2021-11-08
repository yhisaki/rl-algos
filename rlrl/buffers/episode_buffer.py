from rlrl.utils.transpose_list_dict import transpose_list_dict


class EpisodeBuffer(object):
    def __init__(self) -> None:
        super().__init__()
        self.memory: list[list[list[dict]]] = []
        self.length = 0

    def append(self, id: int, state, next_state, action, reward, terminal, reset, **kwargs):
        transition = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminal=terminal,
            reset=reset,
            **kwargs,
        )
        while len(self.memory) < id + 1:
            self.memory.append([[]])
        self.memory[id][-1].append(transition)
        self.length += 1
        if reset:
            self.memory[id].append([])

    def get_episodes(self):
        def _flattening(memory):
            for episodes in memory:
                for episode in episodes:
                    if len(episode) > 0:
                        yield episode

        mem_flatten = list(_flattening(self.memory))
        episodes = [transpose_list_dict(epi) for epi in mem_flatten]
        return episodes

    def __len__(self):
        return self.length

    def clear(self):
        self.memory = []
        self.length = 0
