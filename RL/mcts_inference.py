import numpy as np
import torch
from ZW_utils import std_classes
from ZW_model import GPT
from ZW_Opt import *
from split_functions import (
    bound_creation,
    layout_to_string_single_1d,
    equipments_to_strings,
)
from thermo_validity import *
from ZW_model import GPT

model = GPT(12, 32, 4, 2, 22, 0.1)
model.load_state_dict(torch.load("GPT_NA_psitest/M2_model_8.pt"))
classes = std_classes

layouts = np.load("M2_data_300_8_augmented_layouts.npy", allow_pickle=True)
results = np.load("M2_data_300_8_augmented_results.npy", allow_pickle=True)
print(len(layouts), len(results))
layouts = equipments_to_strings(layouts, classes)
results = 1 - (results - 125) / 175
indices = np.argsort(results)
sorted_results = np.array(results)[indices]
sorted_layouts = np.array(layouts)[indices]
unique, indices = np.unique(sorted_layouts, return_index=True)
unique_results = sorted_results[indices]
unique_layouts = sorted_layouts[indices]
print(len(unique_layouts), len(unique_results))
layouts = unique_layouts.tolist()
results = unique_results
new_layouts = []
new_results = []
found_values = []


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

    def get_next_state(self, state, action):
        column = np.where(state == -1)[0][0]
        state[column] = action
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == -1).astype(np.uint8)

    def check_win(self, state, action):
        if action == 11:
            return True
        return False

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            value = evaluation(self.get_encoded_state(state))
            found_values.append(value)
            return value, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return -1, True
        # -1, True
        return 0, False
        # -1, False

    def get_encoded_state(self, state):
        '''"if the design is less than 23 equipment, return the state up to the last column with a -1
        else return the state as is"'''
        try:
            column = np.where(state == -1)[0][0]
        except:
            column = self.column_count
        encoded_state = state[:column]
        return encoded_state


def evaluation(layout):
    # 1. One hot encoding from integer
    layout = layout.astype(int)
    stringlist = [
        layout_to_string_single_1d(layout),
    ]
    valid_string = validity(stringlist)
    if len(valid_string) == 0:
        return -1

    if valid_string[0] in new_layouts:
        return new_results[new_layouts.index(valid_string[0])]
    if valid_string[0] in layouts:
        return results[layouts.index(valid_string[0])]
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
        if a.result < 300:
            # standardization between 125 and 300 to 1 and 0
            value = 1 - (a.result - 125) / 175
            print(value, valid_string[0])
            new_layouts.append(valid_string[0])
            new_results.append(value)
        else:
            value = -0.25
    except:
        value = -0.5
    return value


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
        # -0.1

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
                child_state = self.game.get_next_state(child_state, action)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
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
        valid_moves = np.ones(self.game.action_size)
        valid_moves[0], valid_moves[6], valid_moves[8], valid_moves[10] = 0, 0, 0, 0
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        for search in range(self.args["num_searches"]):
            node = root
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
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
                value = 0
                node.expand(policy)
            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


fw = Flowsheet()
args = {
    "C": 2,
    "num_searches": 50,
    "dirichlet_epsilon": 0.0,
    "dirichlet_alpha": 0.3,
}
mcts = MCTS(fw, args, model)
# policy = mcts.model(
#             torch.tensor(mcts.game.get_encoded_state(state),dtype=torch.long).unsqueeze(0))
# print(policy)
# policy = F.softmax(policy,dim=-1).squeeze(0).detach().numpy()
# print(policy)
# valid_moves = fw.get_valid_moves(state)
# print(valid_moves)
best_value = 0
generated_designs = []
for i in range(3000):
    state = fw.get_initial_state()
    while True:
        mcts_probs = mcts.search(state)
        action = np.argmax(mcts_probs)
        state = fw.get_next_state(state, action)
        value, is_terminal = fw.get_value_and_terminated(state, action)

        if is_terminal:
            if value > best_value:
                best_value = value
            if i % 50 == 0:
                print(i, "Best Value Found:", best_value)
            generated_designs.append((value, fw.get_encoded_state(state)))
            break

generated_designs = np.array(generated_designs, dtype=object)
np.save(
    f"MCTS_generated_designs_{args['C']}_{args['num_searches']}_{args['dirichlet_epsilon']}_{args['dirichlet_alpha']}.npy",
    generated_designs,
)
np.save("found_values.npy", found_values)
