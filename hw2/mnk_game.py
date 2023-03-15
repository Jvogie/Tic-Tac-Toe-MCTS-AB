import math
import random

import numpy as np
import numpy.typing as npt

from hw2.utils import utility, successors, Node, Tree, GameStrategy


"""
Alpha Beta Search
"""


def max_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the max value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    if utility(state, k) is not None:
        return utility(state,k), None

    v = float(-10000)
    new_successor = None

    for child in successors(state, player="X"):
        value, a2 =min_value(child, alpha, beta, k)
        if value > v:
            v=value
            new_successor= child
        if v >= beta:
            return v, new_successor
        alpha = max(alpha, v)

    return v, new_successor


def min_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the min value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    if utility(state, k) is not None:
        return utility(state, k), None

    v = float(10000)
    new_successor = None

    for child in successors(state, player="O"):
        value, a2 = max_value(child, alpha, beta, k)
        if value < v:
            v = value
            new_successor = child
        if v <= alpha:
            return v, new_successor
        beta = min(beta, v)

    return v, new_successor


"""
Monte Carlo Tree Search
"""


def select(tree: "Tree", state: npt.ArrayLike, k: int, alpha: float):
    """Starting from state, find a terminal node or node with unexpanded
    children. If all children of a node are in tree, move to the one with the
    highest UCT value.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
        alpha (float): exploration parameter
    Returns:
        np.ndarray: the game board state
    """

    # TODO:

    UCT = -math.inf

    while True:
        #finding current node from state and what player it is
        node = tree.get(state)
        player = node.player

        #check to see if terminal node
        if utility(state,k) is not None:
            return state

        #find successor states of current state and check if any arent a node in tree yet
        child_list= successors(state, player)

        for s in child_list:
            if tree.get(s) is None:
                return state

        #check any child node to see if it has a different parent than current node. If so remove
        # from states to add nodes of

        new_child_list =[]
        for b in child_list:
            cnode=tree.get(b)
            if np.array_equal(cnode.parent.state, node.state) is True:
                new_child_list.append(b)

        # take list of states that aren't terminal, nor have been added as node to tree and compute
        # UCT value of each element in list and keep the state with the largest UCT value.
        # That state will be then become new current node and then select function repeats.
        for child in new_child_list:
            child_node = tree.get(child)
            UCT_value = (child_node.w / child_node.N) + alpha * math.sqrt(math.log(child_node.parent.N) / child_node.N)
            if UCT_value >= UCT:
                UCT = UCT_value
                state=child


def expand(tree: "Tree", state: npt.ArrayLike, k: int):
    """Add a child node of state into the tree if it's not terminal and return
    tree and new state, or return current tree and state if it is terminal.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
    Returns:
        tuple[utils.Tree, np.ndarray]: the tree and the game state
    """

    # TODO:

    node=tree.get(state)

    player = node.player

    if player == "X":
        next_player = "O"
    else:
        next_player = "X"

    if utility(state, k) is not None:
        return tree, state

    child_state = successors(state, player)

    if all(tree.get(s) is not None for s in child_state) is True:
        return tree, state

    for child in child_state:
        if tree.get(child) is None:

            new_node = Node(state= child, parent= node, player= next_player)
            tree.add(new_node)
            state=child
            return tree, state



def simulate(state: npt.ArrayLike, player: str, k: int):
    """Run one game rollout from state to a terminal state using random
    playout policy and return the numerical utility of the result.

    Args:
        state (np.ndarray): the game board state
        player (string): the player, `O` or `X`
        k (int): the number of consecutive marks
    Returns:
        float: the utility
    """

    # TODO:
    current_player = player
    while utility(state, k) is None:
        move_list = successors(state, current_player)
        move = random.choice(move_list)
        state = move
        if current_player == "X":
            current_player ="O"
        elif current_player == "O":
            current_player = "X"
    #print(5)
    return utility(state,k)


def backprop(tree: "Tree", state: npt.ArrayLike, result: float):
    """Backpropagate result from state up to the root.
    All nodes on path have N, number of plays, incremented by 1.
    If result is a win for a node's parent player, w is incremented by 1.
    If result is a draw, w is incremented by 0.5 for all nodes.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        result (float): the result / utility value

    Returns:
        utils.Tree: the game tree
    """

    # TODO:
    node = tree.get(state)

    while node is not None:
        node.N +=1
        if node.player =="X" and result == -1:
            node.w +=1
        elif node.player =="O" and result == 1:
            node.w += 1
        elif result == 0:
            node.w += 0.5

        node = node.parent
    #print(6)
    return tree


# ******************************************************************************
# ****************************** ASSIGNMENT ENDS *******************************
# ******************************************************************************


def MCTS(state: npt.ArrayLike, player: str, k: int, rollouts: int, alpha: float):
    # MCTS main loop: Execute MCTS steps rollouts number of times
    # Then return successor with highest number of rollouts
    tree = Tree(Node(state, None, player, 0, 1))

    for i in range(rollouts):
        leaf = select(tree, state, k, alpha)
        tree, new = expand(tree, leaf, k)
        result = simulate(new, tree.get(new).player, k)
        tree = backprop(tree, new, result)

    nxt = None
    plays = 0

    for s in successors(state, tree.get(state).player):
        if tree.get(s).N > plays:
            plays = tree.get(s).N
            nxt = s

    return nxt


def ABS(state: npt.ArrayLike, player: str, k: int):
    # ABS main loop: Execute alpha-beta search
    # X is maximizing player, O is minimizing player
    # Then return best move for the given player
    if player == "X":
        value, move = max_value(state, -float("inf"), float("inf"), k)
    else:
        value, move = min_value(state, -float("inf"), float("inf"), k)

    return value, move


def game_loop(
    state: npt.ArrayLike,
    player: str,
    k: int,
    Xstrat: GameStrategy = GameStrategy.RANDOM,
    Ostrat: GameStrategy = GameStrategy.RANDOM,
    rollouts: int = 0,
    mcts_alpha: float = 0.01,
    print_result: bool = False,
):
    # Plays the game from state to terminal
    # If random_opponent, opponent of player plays randomly, else same strategy as player
    # rollouts and alpha for MCTS; if rollouts is 0, ABS is invoked instead
    current = player
    while utility(state, k) is None:
        if current == "X":
            strategy = Xstrat
        else:
            strategy = Ostrat

        if strategy == GameStrategy.RANDOM:
            state = random.choice(successors(state, current))
        elif strategy == GameStrategy.ABS:
            _, state = ABS(state, current, k)
        else:
            state = MCTS(state, current, k, rollouts, mcts_alpha)

        current = "O" if current == "X" else "X"

        if print_result:
            print(state)

    return utility(state, k)
