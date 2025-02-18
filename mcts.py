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
from tqdm.notebook import trange

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
        return -100
    ohe = np.zeros((len(layout), len(classes)), dtype=object)
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
        if a.result < 1e6:
            value = a.result
            print(valid_string, value)
        else:
            value = -5
    except:
        value = -10
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
        try:
            column = np.where(state == -1)[0][0]
        except:
            column = self.column_count
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
        return player

    def get_opponent_value(self, value):
        return value

    def change_perspective(self, state, player):
        return state

    def get_encoded_state(self, state):
        try:
            column = np.where(state == -1)[0][0]
        except:
            column = self.column_count
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

        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size)
        # all moves are valid if we are not masking valid_moves = self.game.get_valid_moves(state)
        valid_moves = np.ones(self.game.action_size)
        valid_moves[0], valid_moves[6], valid_moves[8], valid_moves[10] = 0, 0, 0, 0
        policy *= valid_moves
        policy /= np.sum(policy)
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
                valid_moves = np.ones(self.game.action_size)
                valid_moves[0], valid_moves[6], valid_moves[8], valid_moves[10] = (
                    0,
                    0,
                    0,
                    0,
                )
                policy = policy * valid_moves
                policy /= np.sum(policy)
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


class Alphazero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            action_probs = self.mcts.search(state)
            memory.append((state, action_probs, player))
            # Temperature lim 0 exploiation, lim inf exploration (more randomness)
            temperature_action_probs = action_probs ** (1 / self.args["temperature"])
            temperature_action_probs /= np.sum(temperature_action_probs)

            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = (
                        value
                        if hist_player == player
                        else self.game.get_opponent_value(value)
                    )
                    returnMemory.append(
                        (
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome,
                        )
                    )
                return returnMemory
            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[
                batchIdx : min(len(memory) - 1, batchIdx + self.args["batch_size"])
            ]
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )
            state = torch.tensor(state).float()
            policy_targets = torch.tensor(policy_targets).float()
            value_targets = torch.tensor(value_targets).float()

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            memory = []

            for selfPlay_iteration in trange(self.args["num_selfPlay_iterations"]):
                memory += self.selfPlay()
            self.model.train()
            for epoch in trange(self.args["num_epochs"]):
                self.train(memory)

            save_path = self.args["save_path"]
            torch.save(
                self.model.state_dict(), f"{save_path}/model_{iteration}_{self.game}.pt"
            )
            torch.save(
                self.optimizer.state_dict(),
                f"{save_path}/optimizer_{iteration}_{self.game}.pt",
            )


game = Flowsheet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# args = {
#     "C":2,
#     "num_searches":60,
#     "num_iterations":3,
#     "num_selfPlay_iterations":500,
#     "num_epochs":4,
#     "batch_size":64,
#     "temperature":1.25,
#     "dirichlet_epsilon":0.25,
#     "dirichlet_alpha":0.3
# }
args = {
    "C": 2,
    "num_searches": 500,
    "num_iterations": 4,
    "num_selfPlay_iterations": 100,
    "num_epochs": 4,
    "batch_size": 128,
    "temperature": 1.25,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.3,
    "save_path": "./RL/policy_model_trials",
}
alphazero = Alphazero(model, optimizer, game, args)
alphazero.learn()
