import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from ZW_utils import std_classes
from ZW_model import GPT
from ZW_Opt import *
from split_functions import (
    bound_creation,
    layout_to_string_single_1d,
    equipments_to_strings,
)
from thermo_validity import *
from tqdm.notebook import trange

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


def evaluation(layout):
    reward = 0
    layout = layout.astype(int)

    # no equipment repetition back to back
    for i in range(1, len(layout) - 2):
        if layout[i] != layout[i + 1]:
            reward += 0.1
    if layout[1] != layout[-2]:
        reward += 0.1
    # has an ending token
    if layout[-1] == 11:
        reward += 0.1
    # has the basic structure [1, 2, 3, 4]
    if 1 in layout and 2 in layout and 3 in layout and 4 in layout:
        reward += 0.4
    # if it has a hx, it has 2 of them
    if 5 in layout:
        if np.count_nonzero(layout == 5) == 2:
            reward += 0.2
    # if it has a splitter, it has 1 of them
    if 9 in layout:
        if np.count_nonzero(layout == 9) == 1:
            reward += 0.1
        if np.count_nonzero(layout == 7) == 2:
            reward += 0.2
    # validity check
    stringlist = [
        layout_to_string_single_1d(layout),
    ]
    valid_string = validity(stringlist)
    reward = reward + 1 if len(valid_string) > 0 else reward - 1

    if len(valid_string) == 0:
        # scale validitiy rewards -1,4 to -1,0
        reward = -1 + (reward - (-1)) * (0 - (-1)) / (4 - (-1))
        return reward

    if valid_string[0] in new_layouts:
        value = new_results[new_layouts.index(valid_string[0])]
        return value
    if valid_string[0] in layouts:
        value = results[layouts.index(valid_string[0])]
        return value

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
        if a.result < 300:
            # standardization between 125 and 300 to 1 and 0.5
            value = 1 - (a.result - 125) * (1 - 0.5) / 175
            new_layouts.append(valid_string[0])
            new_results.append(value)
        elif a.result < 1e6:
            # standardization between 300 and 1e6 to 0.5 and 0.25
            value = 0.5 - (a.result - 300) * (0.5 - 0.25) / (1e6 - 300)
            new_layouts.append(valid_string[0])
            new_results.append(value)
        else:
            value = 0 + (reward - (-1)) * (0.25 - 0) / (4 - (-1))
    except:
        value = 0
    return value


class Flowsheet:
    def __init__(self) -> None:
        self.column_count = 22
        self.action_size = 12
        self.padding_idx = 12
        self.ending_idx = 11

    def __repr__(self):
        return "Flowsheet"

    def get_initial_state(self):
        blank_state = np.ones(self.column_count) * self.padding_idx
        blank_state[0] = 0
        return blank_state

    def get_next_state(self, state, action):
        column = np.where(state == self.padding_idx)[0][0]
        state[column] = action
        return state

    def get_valid_moves(self, state):
        return state.reshape(-1) == self.padding_idx

    def check_win(self, state, action):
        if action == self.ending_idx:
            return True
        return False

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            value = evaluation(self.get_encoded_state(state))
            return value, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return -1, True
        return 0, False

    def get_encoded_state(self, state):
        try:
            column = np.where(state == self.padding_idx)[0][0]
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
        self.valid_moves = np.ones(self.game.action_size)
        (
            self.valid_moves[0],
            self.valid_moves[6],
            self.valid_moves[8],
            self.valid_moves[10],
        ) = (0, 0, 0, 0)

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        lengths = torch.tensor(
            len(self.game.get_encoded_state(state)), dtype=torch.long
        ).reshape(1)
        x = torch.tensor(state).reshape(1, -1, 1)
        policy, _ = self.model(x, lengths)
        policy = torch.softmax(policy, axis=-1).squeeze(0).detach().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size)

        valid_move = self.valid_moves.copy()
        valid_move[int(x.squeeze()[lengths - 1].item())] = 0
        policy *= valid_move
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
                lengths = torch.tensor(
                    len(self.game.get_encoded_state(node.state)), dtype=torch.long
                ).reshape(1)
                x = torch.tensor(node.state).reshape(1, -1, 1)
                policy, value = self.model(x, lengths)
                policy = torch.softmax(policy, axis=-1).squeeze(0).detach().numpy()
                valid_move = self.valid_moves.copy()
                valid_move[int(x.squeeze()[lengths - 1].item())] = 0
                policy *= valid_move
                policy /= np.sum(policy)
                # step value penalty of -0.1
                value = value.item() - 0.1
                node.expand(policy)
            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)

        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


class Alphazero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        state = self.game.get_initial_state()

        while True:
            action_probs = self.mcts.search(state)
            memory.append((state.copy(), action_probs.copy()))
            # Temperature lim 0 exploiation, lim inf exploration (more randomness)
            temperature_action_probs = action_probs ** (1 / self.args["temperature"])
            temperature_action_probs /= np.sum(temperature_action_probs)

            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            state = self.game.get_next_state(state, action)
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            if is_terminal:
                returnMemory = []
                for hist_state, hist_action_probs in memory:
                    hist_outcome = value
                    returnMemory.append(
                        (
                            hist_state,
                            hist_action_probs,
                            hist_outcome,
                        )
                    )
                return returnMemory

    def train(self, memory):
        random.shuffle(memory)
        epoch_loss = 0
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[
                batchIdx : min(len(memory) - 1, batchIdx + self.args["batch_size"])
            ]
            states, policy_targets, value_targets = zip(*sample)
            encoded_states = [self.game.get_encoded_state(s) for s in states]
            lengths = torch.tensor([len(s) for s in encoded_states])
            states = np.array(states).reshape(-1, self.game.column_count, 1)
            # padding to the maximum length
            states, policy_targets, value_targets = (
                np.array(states),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )
            states = torch.tensor(states)
            policy_targets = torch.tensor(policy_targets)
            value_targets = torch.tensor(value_targets).float()

            out_policy, out_value = self.model(states, lengths)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        print("Epoch Loss:", epoch_loss / len(memory))

    def learn(self):
        for iteration in range(
            self.args["starting_iteration"], self.args["num_iterations"]
        ):
            print(f"Iteration {iteration} of {self.args['num_iterations']}")
            memory = []

            for selfPlay_iteration in trange(self.args["num_selfPlay_iterations"]):
                memory += self.selfPlay()
                if selfPlay_iteration % 50 == 0:
                    print(
                        f"Self Play Iteration {selfPlay_iteration}:",
                        len(memory),
                        "samples",
                    )
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
            np.save(f"{save_path}/new_layouts.npy", new_layouts)
            np.save(f"{save_path}/new_results.npy", new_results)
            # 5 example inference
            for i in range(5):
                state = self.game.get_initial_state()
                while True:
                    mcts_probs = self.mcts.search(state)
                    action = np.argmax(mcts_probs)
                    state = self.game.get_next_state(state, action)
                    value, is_terminal = self.game.get_value_and_terminated(
                        state, action
                    )
                    if is_terminal:
                        print(value, self.game.get_encoded_state(state))
                        break


class LSTM_packed(nn.Module):
    def __init__(self, embd_size, hidden_size):
        super(LSTM_packed, self).__init__()
        self.embedding = nn.Embedding(13, embd_size, padding_idx=12)
        self.lstm = nn.LSTM(
            embd_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1
        )
        self.valuehead = nn.Linear(hidden_size, 1)
        self.policyhead = nn.Linear(hidden_size, 12)

    def forward(self, x, lengths):
        x = self.embedding(x.long())
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        output, (hidden, _) = self.lstm(x)
        value = self.valuehead(hidden[-1])
        policy = self.policyhead(hidden[-1])
        return policy, value


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, out_features=12):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.policyhead = nn.Linear(in_features=hidden_size, out_features=out_features)
        self.valuehead = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x, lengths):
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        output, (hidden, _) = self.lstm(x_packed.float())
        policy = self.policyhead(hidden[-1])
        value = self.valuehead(hidden[-1])
        return policy, value


class LSTMemb(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2, out_features=12, emb_size=16):
        super().__init__()
        self.embedding = nn.Embedding(13, embedding_dim=emb_size, padding_idx=12)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.policyhead = nn.Linear(in_features=hidden_size, out_features=out_features)
        self.valuehead = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x, lengths):
        x = self.embedding(x.squeeze(-1).long())
        # x_packed = nn.utils.rnn.pack_padded_sequence(
        #     x, lengths, batch_first=True, enforce_sorted=False
        # )
        output, (hidden, _) = self.lstm(x)
        policy = self.policyhead(hidden[-1])
        value = self.valuehead(hidden[-1])
        return policy, value


if __name__ == "__main__":
    new_layouts = np.load(
        "RL/policy_value_model_trials/new_layouts.npy", allow_pickle=True
    ).tolist()
    new_results = np.load(
        "RL/policy_value_model_trials/new_results.npy", allow_pickle=True
    ).tolist()
    game = Flowsheet()
    # # model = LSTM_packed(64, 256)
    model = LSTMemb(hidden_size=256, emb_size=32)
    model.load_state_dict(
        torch.load("RL/policy_value_model_trials/model_14_Flowsheet.pt")
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, weight_decay=1e-4)
    args = {
        "C": 1.5,
        "num_searches": 20,
        "num_iterations": 25,
        "num_selfPlay_iterations": 500,
        "num_epochs": 5,
        "batch_size": 64,
        "temperature": 1.5,
        "dirichlet_epsilon": 0.3,
        "dirichlet_alpha": 0.3,
        "starting_iteration": 15,
        "save_path": "./RL/policy_value_model_trials",
    }
    alphazero = Alphazero(model, optimizer, game, args)
    alphazero.learn()

    # inference
    fw = Flowsheet()
    model.load_state_dict(
        torch.load("RL/policy_value_model_trials/model_21_Flowsheet.pt")
    )
    mcts = MCTS(fw, args, model)
    best_value = -np.inf
    generated_designs = []
    for i in range(5):
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
    for v, l in generated_designs:
        print(v, l)
    valid_designs = [
        (value, layout) for value, layout in generated_designs if value > 0
    ]
    for v, l in valid_designs:
        print(v, l)
    print(len(valid_designs))
