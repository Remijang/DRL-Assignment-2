# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.raw = 0
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()
        self.lookup.init()

    class lookup:

        find = [None] * 65536

        class entry:
            def __init__(self, row : int):
                V = [ (row >> 0) & 0x0f, (row >> 4) & 0x0f, (row >> 8) & 0x0f, (row >> 12) & 0x0f ]
                L, score = Game2048Env.lookup.entry.mvleft(V)
                V.reverse()
                R, score = Game2048Env.lookup.entry.mvleft(V)
                R.reverse()
                self.raw = row
                self.left = (L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12)
                self.right = (R[0] << 0) | (R[1] << 4) | (R[2] << 8) | (R[3] << 12)
                self.score = score

            def move_left(self, raw : int, sc : int, i : int):
                return raw | (self.left << (i << 4)), sc + self.score

            def move_right(self, raw : int, sc : int, i : int):
                return raw | (self.right << (i << 4)), sc + self.score

            @staticmethod
            def mvleft(row : int):
                buf = [t for t in row if t]
                res, score = [], 0
                while buf:
                    if len(buf) >= 2 and buf[0] is buf[1]:
                        buf = buf[1:]
                        buf[0] += 1
                        score += 1 << buf[0]
                    res += [buf[0]]
                    buf = buf[1:]
                return res + [0] * (4 - len(res)), score

        @classmethod
        def init(cls) -> None:
            cls.find = [cls.entry(row) for row in range(65536)]


    def fetch(self, i : int) -> int:
        return (self.raw >> (i << 4)) & 0xffff

    def place(self, i : int, r : int) -> None:
        self.raw = (self.raw & ~(0xffff << (i << 4))) | ((r & 0xffff) << (i << 4))

    def at(self, i : int) -> int:
        return (self.raw >> (i << 2)) & 0x0f

    def set(self, i : int, t : int) -> None:
        self.raw = (self.raw & ~(0x0f << (i << 2))) | ((t & 0x0f) << (i << 2))

    def transpose(self) -> None:
        self.raw = (self.raw & 0xf0f00f0ff0f00f0f) | ((self.raw & 0x0000f0f00000f0f0) << 12) | ((self.raw & 0x0f0f00000f0f0000) >> 12)
        self.raw = (self.raw & 0xff00ff0000ff00ff) | ((self.raw & 0x00000000ff00ff00) << 24) | ((self.raw & 0x00ff00ff00000000) >> 24)

    def mirror(self) -> None:
        self.raw = ((self.raw & 0x000f000f000f000f) << 12) | ((self.raw & 0x00f000f000f000f0) << 4) \
                 | ((self.raw & 0x0f000f000f000f00) >> 4) | ((self.raw & 0xf000f000f000f000) >> 12)

    def flip(self) -> None:
        self.raw = ((self.raw & 0x000000000000ffff) << 48) | ((self.raw & 0x00000000ffff0000) << 16) \
                 | ((self.raw & 0x0000ffff00000000) >> 16) | ((self.raw & 0xffff000000000000) >> 48)

    def rotate_clockwise(self) -> None:
        self.transpose()
        self.mirror()

    def rotate_counterclockwise(self) -> None:
        self.transpose()
        self.flip()

    def reset(self):
        self.raw = 0
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.raw

    def add_random_tile(self):
        space = [i for i in range(16) if self.at(i) == 0]
        if space:
            self.set(random.choice(space), 1 if random.random() < 0.9 else 2)

    def move_left(self):
        move = 0
        prev = self.raw
        for i in range(4):
            move, self.score = self.lookup.find[self.fetch(i)].move_left(move, self.score, i)
        self.raw = move
        return move != prev

    def move_right(self):
        move = 0
        prev = self.raw
        for i in range(4):
            move, self.score = self.lookup.find[self.fetch(i)].move_right(move, self.score, i)
        self.raw = move
        return move != prev

    def move_up(self):
        self.rotate_clockwise()
        moved = self.move_right()
        self.rotate_counterclockwise()
        return moved

    def move_down(self):
        self.rotate_clockwise()
        moved = self.move_left()
        self.rotate_counterclockwise()
        return moved

    def is_game_over(self):
        for i in range(16):
          if self.at(i) == 0:
            return False
        for i in range(4):
            for j in range(4):
                if j < 3 and self.at(i * 4 + j) == self.at(i * 4 + j + 1):
                    return False
                if i < 3 and self.at(i * 4 + j) == self.at((i + 1) * 4 + j):
                    return False
        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.raw, self.score, done, {}

    def try_step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        done = self.is_game_over()

        return self.raw, self.score, done, {}

    def is_move_legal(self, action):
        old_raw = self.raw
        old_score = self.score
        old_last_move_valid = self.last_move_valid
        self.step(action)
        self.raw, old_raw = old_raw, self.raw
        self.score, old_score = old_score, self.score
        self.last_move_valid, old_last_move_valid = old_last_move_valid, self.last_move_valid
        return self.raw != old_raw


def rot90(board):
    a = [12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3]
    return [a[i] for i in board]

def rot180(board):
    a = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    return [a[i] for i in board]

def rot270(board):
    a = [3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12]
    return [a[i] for i in board]

def flip_vertical(board):
    a = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    return [a[i] for i in board]

def flip_horizontal(board):
    a = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
    return [a[i] for i in board]



class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [[0.0 for _ in range(16 ** len(patterns[0]))] for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)
        # print(self.patterns)
        # print(self.symmetry_patterns)
        self.env2 = Game2048Env()

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        return [
            pattern,
            rot90(pattern),
            rot180(pattern),
            rot270(pattern),
            flip_vertical(pattern),
            flip_horizontal(pattern),
            flip_vertical(rot90(pattern)),
            flip_horizontal(rot90(pattern)),
        ]
    '''
    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))
    '''
    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        self.env2.raw = board
        idx = 0
        for i in coords:
            idx *= 16
            idx += self.env2.at(i)
        return idx

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        value = 0
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            value += self.weights[i // 8][feature]
        # print(board, value)
        return value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        delta /= len(self.symmetry_patterns)
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            self.weights[i // 8][feature] += alpha * delta

patterns = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 4, 5, 6],
    [1, 2, 5, 6, 9, 13],
    [0, 1, 5, 6, 7, 10],
    [0, 1, 2, 5, 9, 10],
    [0, 1, 5, 9, 13, 14],
    [0, 1, 5, 8, 9, 13],
    [0, 1, 2, 4, 6, 10]
]

import struct

def load_large_vector_from_binary_file(filename):
    try:
        with open(filename, 'rb') as infile:
            # 讀取維度資訊
            dim2 = struct.unpack('<Q', infile.read(8))[0]
            dim3 = struct.unpack('<Q', infile.read(8))[0]

            matrix = []
            for j in range(dim2):
                row_bytes = infile.read(dim3 * 4)
                row = list(struct.unpack('<{}f'.format(dim3), row_bytes))
                matrix.append(row)
        return matrix

    except FileNotFoundError:
        print(f"Error opening file: {filename}")
    except struct.error:
        print("Error reading file format.")


approximator = NTupleApproximator(board_size = 4, patterns = patterns)
approximator.weights = load_large_vector_from_binary_file("approximator")

norm = 275207.4336295339

env3 = Game2048Env()

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, chance, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 1
        self.total_reward = 0.0
        self.chance = chance
        self.expand = 0
        # List of untried actions based on the current state's legal moves
    def expanded(self):
        env3.raw = self.state
        env3.score = self.score
        if not self.chance:
            self.untried_actions = [a for a in range(4) if env3.is_move_legal(a)]
            old_state = env3.raw
            old_score = env3.score
            old_is_move_legal = env3.is_move_legal
            for a in self.untried_actions:
                env3.raw = old_state
                env3.score = old_score
                env3.is_move_legal = old_is_move_legal
                env3.try_step(a)
                self.children[a] = TD_MCTS_Node(env3.raw, env3.score, 1, self, a)
        self.expand = 1


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.env2 = Game2048Env()

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.raw = state
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        if node.chance:
            state = node.state
            self.env2.raw = state
            self.env2.add_random_tile()
            if env.raw not in node.children:
                node.children[self.env2.raw] = TD_MCTS_Node(self.env2.raw, node.score, 0, node)
            return node.children[self.env2.raw]
        else:
            uct_values = []
            for action, child in node.children.items():
                uct_values.append(child.total_reward / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits))
            best_action = list(node.children.keys())[np.argmax(uct_values)]
            return node.children[best_action]

    def rollout(self, sim_env, depth, chance):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        if chance:
            return self.approximator.value(sim_env.raw)

        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        if not legal_moves:
            return self.approximator.value(sim_env.raw)
        V = 0.0
        old_score = sim_env.score
        old_state = sim_env.raw
        old_is_move_legal = sim_env.is_move_legal
        for action in legal_moves:
            sim_env.raw = old_state
            sim_env.score = old_score
            sim_env.is_move_legal = old_is_move_legal
            state, reward, done, _ = sim_env.try_step(action)
            sum = self.approximator.value(sim_env.raw) + reward
            V = V if V > sum else sum
        sim_env.raw = old_state
        sim_env.score = old_score
        sim_env.is_move_legal = old_is_move_legal
        return V

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent

    def run_simulation(self, root):
        node = root
        init_score = node.score

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.expand:
            node = self.select_child(node)

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        sim_env = self.create_env_from_state(node.state, node.score)
        mid_score = sim_env.score
        if not node.expand:
            node.expanded()

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth, node.chance)
        rollout_reward += mid_score - init_score
        rollout_reward /= norm * 64
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = []
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = [action]
            elif child.visits == best_visits:
                best_action.append(action)
        best_action = random.choice(best_action)
        return best_action, distribution

env = Game2048Env()
td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

def get_action(state, score):
    s = 0
    for i in range(4):
        for j in range(4):
            if state[i][j]:
                s |= (int(np.log2(state[i][j])) & 0xf) << ((i * 4 + j) << 2)
    root = TD_MCTS_Node(s, score, 0)

    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    best_act, distrib = td_mcts.best_action_distribution(root)
    return best_act
