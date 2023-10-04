from collections import deque
import enum
from dataclasses import dataclass, field
from copy import deepcopy

import torch_geometric.data as gd
import torch
import networkx as nx
import numpy as np

from rdkit import Chem
import matplotlib.pyplot as plt


class NodeType(enum.Enum):
    C = enum.auto()
    N = enum.auto()
    O = enum.auto()
    S = enum.auto()
    P = enum.auto()
    F = enum.auto()
    I = enum.auto()
    Cl = enum.auto()
    Br = enum.auto()

    def __repr__(self):
        return self.name


class EdgeType(enum.Enum):
    SINGLE = enum.auto()
    DOUBLE = enum.auto()
    TRIPLE = enum.auto()

    def __repr__(self):
        return self.name


@dataclass
class State:
    node_types: list[NodeType] = field(default_factory=list)
    edge_types: list[EdgeType] = field(default_factory=list)
    edge_list: list[tuple[int, int]] = field(default_factory=list)

    num_node: int = field(default_factory=int, repr=False)
    num_edge: int = field(default_factory=int, repr=False)
    _edge_set: set[tuple[int, int]] = field(default_factory=set, repr=False)

    def __post_init__(self):
        self.num_node = len(self.node_types)
        self.num_edges = len(self.edge_types)
        self._edge_set = set(self.edge_list)
        self._color_map = np.random.default_rng(0).uniform(0, 1, size=(100, 3))

    def add_node(self, node_type: NodeType):
        self.node_types.append(node_type)
        self.num_node += 1

    def add_edge(self, i: int, j: int, edge_type: EdgeType):
        edge = (i, j) if i < j else (j, i)
        self.edge_types.append(edge_type)
        self.edge_list.append(edge)
        self._edge_set.add(edge)
        self.num_edge += 1

    def get_non_edge_list(self):
        num_nodes = len(self.node_types)
        non_edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge = (i, j)
                if edge not in self._edge_set:
                    non_edges.append(edge)
        return non_edges

    @classmethod
    def from_mol(cls, mol: Chem.RWMol):
        mol = deepcopy(mol)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

        node_types = []
        edge_types = []
        edge_list = []
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            node_types += [getattr(NodeType, symbol)]

        for bond in mol.GetBonds():
            bt = str(bond.GetBondType())
            edge_types += [getattr(EdgeType, bt)]
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list.append((i, j))
        return cls(node_types, edge_types, edge_list)

    @classmethod
    def from_smiles(cls, smi: str):
        mol = Chem.MolFromSmiles(smi)
        return cls.from_mol(mol)

    def to_mol(self):
        m = Chem.RWMol()
        for nt in self.node_types:
            m.AddAtom(Chem.Atom(nt.name))
        for i, (u, v) in enumerate(self.edge_list):
            et = self.edge_types[i]
            order = Chem.BondType.names[et.name]
            m.AddBond(u, v, order=order)
        return m

    def viz(self, figsize=(3, 3), with_labels=True):
        graph = nx.from_edgelist(self.edge_list)
        for i in range(self.num_node):
            graph.add_node(i)
        node_color = [tuple(self._color_map[x.value]) for x in self.node_types]
        edge_color = [tuple(self._color_map[x.value]) for x in self.edge_types]

        plt.figure(figsize=figsize)
        return nx.draw(
            graph,
            node_color=node_color,
            edge_color=edge_color,
            with_labels=with_labels,
        )


class ActionType(enum.Enum):
    AddNode = enum.auto()
    AddEdge = enum.auto()
    STOP = enum.auto()


@dataclass
class Action:
    type: ActionType = None
    source: int = None
    target: int = None
    node_type: NodeType = None
    edge_type: EdgeType = None

    def is_sane(self):
        if self.type == ActionType.AddNode:
            assert self.node_type is not None

        elif self.type == ActionType.AddEdge:
            assert self.source is not None
            assert self.target is not None
            assert self.edge_type is not None

        else:
            raise ValueError(f"Action type `{self.type}` encountered")


class Trajectory:
    def __init__(self, states, actions):
        assert len(states) == len(actions)
        self.states = states
        self.actions = actions

    @property
    def last_state(self):
        return self.states[-1]

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return f"{self.__class__.__name__}(size: {len(self)})"


num_node_types = len(NodeType)
num_edge_types = len(EdgeType)


def initial_state():
    return State([NodeType.C])


def step(state: State, action: Action):
    next_state, done = deepcopy(state), False

    if action.type == ActionType.AddNode:
        next_state.add_node(action.node_type)

    elif action.type == ActionType.AddEdge:
        next_state.add_edge(action.source, action.target, action.edge_type)

    else:  # ActionType.STOP
        done = True

    return next_state, done


def Action_to_idx(state: State, action: Action) -> int:
    y = None

    if action is not None:
        if action.type == ActionType.STOP:
            y = 0

        elif action.type == ActionType.AddNode:
            y = action.node_type.value

        elif action.type == ActionType.AddEdge:
            non_edges = state.get_non_edge_list()
            i, j = (action.source, action.target)
            edge = (i, j) if i < j else (j, i)
            y = (
                num_node_types
                + num_edge_types * non_edges.index(edge)
                + action.edge_type.value
            )

        else:
            raise ValueError("Invalid action type")

    return y


def idx_to_Action(state: State, idx: int) -> Action:
    action = None
    if idx == 0:
        action = Action(ActionType.STOP)

    elif idx - 1 < num_node_types:
        action = Action(type=ActionType.AddNode, node_type=NodeType(idx))
    else:
        idx = idx - 1 - num_node_types
        i = idx // num_edge_types
        t = idx % num_edge_types
        non_edges = state.get_non_edge_list()
        src, tgt = non_edges[i]
        action = Action(
            type=ActionType.AddEdge,
            source=src,
            target=tgt,
            edge_type=EdgeType(t + 1),
        )

    return action


def to_Data(state: State) -> gd.Data:
    # edge_index
    edge_index = (
        torch.LongTensor(
            [e for i, j in state.edge_list for e in [(i, j), (j, i)]],
        )
        .reshape(-1, 2)
        .t()
        .contiguous()
    )

    # types
    node_type = torch.LongTensor([n.value for n in state.node_types]) - 1
    edge_type = (
        torch.LongTensor([e for n in state.edge_types for e in (n.value, n.value)]) - 1
    )

    # non_edge_index
    non_edges = state.get_non_edge_list()
    non_edge_index = torch.LongTensor(non_edges).reshape(-1, 2).t().contiguous()
    num_non_edges = non_edge_index.shape[1]

    return gd.Data(
        edge_index=edge_index,
        non_edge_index=non_edge_index,
        node_type=node_type,
        edge_type=edge_type,
        num_non_edges=num_non_edges,
    )


def get_bfs_trajectory(state):
    # We start from the node with type C
    for init, n in enumerate(state.node_types):
        if n == NodeType.C:
            break

    cur = initial_state()
    queue = deque([init])
    label_map = [-1] * state.num_node
    label_map[init] = 0

    neighbor = [[] for _ in range(state.num_node)]
    for i, (u, v) in enumerate(state.edge_list):
        t = state.edge_types[i]
        neighbor[u].append((v, t))
        neighbor[v].append((u, t))

    actions = []
    states = [cur]

    def take_action(action):
        next_cur, *_ = step(cur, action)
        states.append(next_cur)
        actions.append(action)
        return next_cur

    while queue:
        u = queue.popleft()
        for v, edge_type in neighbor[u]:
            if label_map[v] == -1:  # first visit
                label_map[v] = cur.num_node
                queue.append(v)

                # add node
                cur = take_action(
                    Action(type=ActionType.AddNode, node_type=state.node_types[v])
                )
                # connect edge to the added node
                cur = take_action(
                    action=Action(
                        type=ActionType.AddEdge,
                        edge_type=edge_type,
                        source=label_map[u],
                        target=label_map[v],
                    )
                )

            else:  # second visit
                i, j = label_map[u], label_map[v]
                i, j = (i, j) if i < j else (j, i)
                if (i, j) not in cur._edge_set:
                    # add edge
                    cur = take_action(
                        Action(
                            type=ActionType.AddEdge,
                            edge_type=edge_type,
                            source=i,
                            target=j,
                        )
                    )

    actions.append(Action(ActionType.STOP))
    return Trajectory(states, actions)
