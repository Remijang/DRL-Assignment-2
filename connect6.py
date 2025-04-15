import sys
import numpy as np
import random
import copy
import math


class Node:
    def __init__(self, board, parent=None, action=None, size=19):
        self.size = size
        self.board = copy.deepcopy(board)
        self.action = action
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.expand = 0
        self.untried_positions = self.get_legal_positions()

    def get_legal_positions(self):
        empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
        return empty_positions

    def full_expand(self):
        return len(self.untried_positions) == 0

class MCTS:
    def __init__(self, size=19, iteration=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.size = size
        self.iteration = iteration
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.env = Connect6Game(self)

    def create_env_from_state(self, board):
        new_env = copy.deepcopy(self.env)
        new_env.board = copy.deepcopy(board)
        return new_env

    def select_child(self, node):
        uct_values = []
        for action, child in node.children.items():
            uct_values.append(child.total_reward / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits))
        best_action = list(node.children.keys())[np.argmax(uct_values)]
        return node.children[best_action]

    def rollout(self, sim_env, depth, node):
        r, c, my_turn = node.action
        for _ in range(depth):
            new_turn = sim_env.get_turn()
            empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if sim_env.board[r, c] == 0]
            r, c = random.sample(empty_positions, 1)[0]
            sim_env.board[r, c] = new_turn
        op_turn = 3 - my_turn
        op_score = sim_env.turn_evaluate(op_turn, (r, c))
        my_score = sim_env.turn_evaluate(my_turn, (r, c))
        return op_score if op_score > my_score else my_score

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent
    
    def run_simulation(self, root, turn):
        node = root
        while node.full_expand() and node.children:
            node = self.select_child(node)
        sim_env = self.create_env_from_state(node.board)
        if node.untried_positions:
            pos = node.untried_positions.pop()
            r, c = pos
            turn = sim_env.get_turn()
            sim_env.board[r, c] = turn
            child = Node(sim_env.board, node, (r, c, turn))
            node.children[(r, c, turn)] = child
            node = child
        rollout_reward = self.rollout(sim_env, self.rollout_depth, node)
        self.backpropagate(node, rollout_reward)

    def best_action(self, root):
        best_visits = -1
        best_action = []
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = [action]
            elif child.visits == best_visits:
                best_action.append(action)
        best_action = random.choice(best_action)
        return best_action

class Connect6Game:
    def __init__(self, mcts, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.mcts = mcts

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def turn_evaluate(self, turn, pos):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        lut = {0: 0, 1: 1, 2: 4, 3: 16, 4: 64, 5: 256, 6: 1024}
        ret = 0.0
        pos_r, pos_c = pos
        self.board[pos_r, pos_c] = turn
        op_turn = 3 - turn
        for r in range(self.size):
            for c in range(self.size):
                for dr, dc in directions:
                    count = [0, 0, 0]
                    rr, cc = r, c
                    total = 0
                    for _ in range(6):
                        if 0 <= rr < self.size and 0 <= cc < self.size:
                            count[self.board[rr, cc]] += 1
                            total += 1
                            rr += dr
                            cc += dc
                        else:
                            break
                    if total == 6 and count[op_turn] == 0:
                        ret += lut[count[turn]]
        return ret

    def get_turn(self):
        b_cnt, w_cnt = 0, 0
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 1:
                    b_cnt += 1
                elif self.board[r, c] == 2:
                    w_cnt += 1
        if w_cnt % 2 == 0 and w_cnt >= b_cnt:
            turn = 1
        else:
            turn = 2
        return turn

    def generate_move(self, color):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return

        turn = 1 if color == 'B' else 2
        root = Node(self.board)
        for _ in range(mcts.iteration):
            mcts.run_simulation(root, turn)

        best_action = mcts.best_action(root)[:2]
        selected = [best_action]
        move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
        
        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)
        return

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    mcts = MCTS(size=19, iteration=10000, exploration_constant=1.41, rollout_depth=0, gamma=1)
    game = Connect6Game(mcts=mcts)
    game.run()
