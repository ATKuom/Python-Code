import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from ZW_utils import std_classes
from ZW_model import GPT
from ZW_Opt import *
from split_functions import bound_creation, layout_to_string_single_1d
from thermo_validity import *

model = GPT(12, 32, 4, 2, 22, 0.1)
model.load_state_dict(torch.load("GPT_NA_psitest/M1_model_10.pt"))
classes = std_classes


def evaluation(layout):
    # 1. One hot encoding from integer
    layout = layout.astype(int)
    stringlist = [
        layout_to_string_single_1d(layout),
    ]
    valid_string = validity(stringlist)
    if len(valid_string) == 0:
        return -1e6
    ohe = np.zeros((len(layout), len(classes)))
    for i, l in enumerate(layout):
        ohe[i, l] = 1
    equipment, bounds, x, splitter = bound_creation(ohe)
    swarmsize_factor = 7
    nv = len(bounds)
    particle_size = swarmsize_factor * nv
    if 5 in equipment:
        particle_size += -1 * swarmsize_factor
    if 9 in equipment:
        particle_size += -2 * swarmsize_factor
    iterations = 30
    try:
        a = PSO(objective_function, bounds, particle_size, iterations, nv, equipment)
        value = -1 * a.result
    except:
        value = -1e6
    return value


class Flowsheet:
    def __init__(self) -> None:
        self.column_count = 23
        self.action_size = 12

    def __repr__(self):
        return "Flowsheet"

    def get_initial_state(self):
        blank_state = np.ones(self.column_count) * -1
        blank_state[0] = 0
        return blank_state

    def get_next_state(self, state, action, player=1):
        column = np.where(state == -1)[0][0]
        state[column] = action
        return state

    def get_valid_moves(self, state):
        return state.reshape(-1) == -1

    def check_win(self, state, action):
        if action == None:
            return False
        if action == 11:
            return True
        return False

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            value = evaluation(self.get_encoded_state(state))
            return value, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return value

    def change_perspective(self, state, player):
        return state

    def get_encoded_state(self, state):
        column = np.where(state == -1)[0][0]
        encoded_state = state[:column]
        return encoded_state


class Node:
    def __init__(
        self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0
    ):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            # 1 - because of switching player the child position is the opponent position
            q_value = ((child.value_sum / child.visit_count) + 1) / 2
        return (
            q_value
            + self.args["C"]
            * (np.sqrt(self.visit_count) / (child.visit_count + 1))
            * child.prior
        )

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        value = self.game.get_opponent_value(value)
        if self.parent != None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        # noise addition
        policy = self.model(
            torch.tensor(
                self.game.get_encoded_state(state), dtype=torch.long
            ).unsqueeze(0)
        )[:, -1, :]
        policy = torch.softmax(policy, axis=-1).squeeze(0).detach().numpy()

        # policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
        #     "dirichlet_epsilon"
        # ] * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size)

        # all moves are valid if we are not masking valid_moves = self.game.get_valid_moves(state)
        # policy*=valid_moves
        # policy /= np.sum(policy)
        root.expand(policy)
        for search in range(self.args["num_searches"]):
            # selection
            node = root

            while node.is_fully_expanded():
                node = node.select()
                # some noise to promote exploration

            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(node.state), dtype=torch.long
                    ).unsqueeze(0)
                )[:, -1, :]
                policy = torch.softmax(policy, axis=-1).squeeze(0).detach().numpy()
                # valid_moves = self.game.get_valid_moves(node.state)
                # policy = policy * valid_moves
                # policy /= np.sum(policy)

                # value = value.item()
                value = 0.5
                # expansion
                node.expand(policy)
            # backpropagation
            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        # return visit_counts


ttt = Flowsheet()
player = 1
args = {
    "C": 1.41,
    "num_searches": 10,
}
mcts = MCTS(ttt, args, model)
state = ttt.get_initial_state()
# state = [0, 1, 5, 2, 3, 9, 4, 7, 5, 4, 7, 4, 11, -1]
# state = np.array(state)
# sf = ttt.get_encoded_state(state)
# print(sf)
# val = evaluation(sf)
# print(val)
while True:
    print(state)
    if player == 1:
        valid_moves = ttt.get_valid_moves(state)
        action = int(input("Enter action: "))

    else:
        neutral_state = state
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
    state = ttt.get_next_state(state, action, player)
    value, is_terminal = ttt.get_value_and_terminated(state, action)
    if is_terminal:
        print(state)
        if value == 1:
            print("Player", player, "wins")
        else:
            print("Draw")
        break

    player = ttt.get_opponent(player)
