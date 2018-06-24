from states import State


class StateMapping:
    def __init__(self):
        self.d = {}

    def __setitem__(self, key: State, value):
        self.d[key] = value

    def __getitem__(self, item: State):
        if item not in self.d:
            self.d[item] = 0
            return 0
        else:
            return self.d[item]

    def __str__(self):
        l = []
        for key in sorted(self.d.keys()):
            l.append(str(key) + " : " + "%3.2f" % self.d[key])
        return "\n".join(l)


class StateActionMapping:
    def __init__(self, n: int):
        self.n = n
        self.d = {}

    def __init_action_dict(self):
        return {i: 0 for i in range(self.n)}

    def __setitem__(self, key, value):
        state, action = key
        if state not in self.d:
            self.d[state] = self.__init_action_dict()

        self.d[state][action] = value

    def __getitem__(self, item):
        state, action = item
        if state not in self.d:
            self.d[state] = self.__init_action_dict()

        return self.d[state][action]

    def get_action_utility(self, state):
        if state not in self.d:
            self.d[state] = self.__init_action_dict()

        return self.d[state]

    def __str__(self):
        l = []
        for state in sorted(self.d.keys()):
            for action in sorted(self.d[state].keys()):
                l.append(str(state) + ", " + str(action) + " : " + "%3.2f" % self.d[state][action])
        return "\n".join(l)