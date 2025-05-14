"""Module for creating a labyrinth based on a python file"""
import ast
from typing import Any, List, Tuple, Dict

from gymnasium import Env

import numpy as np
from labyrinth.utils import get_neighbors
from labyrinth.interp import Interpreter


def split(list_a: List[Any], chunk_size: int):
    """Split a list into chunks.

    Args:
        list_a (List[Any]): list to be split
        chunk_size (int): size of the chunks

    Yields:
        Iterator[List[List[Any]]]: chunks of the list
    """
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


def convert(item: int, other: int, width: int) -> int:
    """Convert the position of the nodes into the correct position.

    Args:
        item (int): current position
        other (int): other position
        width (int): width of the labyrinth

    Returns:
        int: new position
    """
    mod = -1 if item > other else +1
    dif = int(width) if abs(item - other) > 1 else 1
    return int(other + (mod * dif))


def get_local_position(position: int, size: int) -> List[int]:
    """Get the local position of a node in the labyrinth.

    Args:
        position (int): global position of the node
        size (int): size of the labyrinth

    Returns:
        List[int]: local position of the node
    """
    column = position // size
    row = position - (column * size)
    return [column, row]


def get_nodes(labyrinth_shape: Tuple[int, int]) -> List[int]:
    """Get all nodes in the labyrinth.

    Args:
        labyrinth_shape (Tuple[int, int]): shape of the labyrinth

    Returns:
        List[int]: list of nodes in the labyrinth
    """
    number_of_nodes = labyrinth_shape[0] * labyrinth_shape[1]
    nodes = list(range(number_of_nodes))
    nodes = list(split(nodes, labyrinth_shape[0]))[::2]
    nodes = [item for sublist in nodes for item in sublist if item % 2 == 0]
    return nodes


def find_edges_start_and_end(
    nodes: List[int],
    vector_labyrinth: List[str],
    labyrinth: List[List[str]],
    labyrinth_original_shape: Tuple[int, int],
    labyrinth_shape: Tuple[int, int]
) -> Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    """Find all edges in the labyrinth.

    Args:
        nodes (List[int]): list of nodes in the labyrinth
        vector_labyrinth (List[str]): vector representation of the labyrinth
        labyrinth (List[List[str]]): labyrinth structure
        labyrinth_original_shape (Tuple[int, int]): original shape of the labyrinth
        labyrinth_shape (Tuple[int, int]): shape of the labyrinth

    Returns:
        Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]: edges, start, end positions
    """
    edges = []
    end = None
    start = None
    for node in nodes:
        if vector_labyrinth[node] == 'S':
            start = nodes.index(node)
            start = get_local_position(start, labyrinth_original_shape[0])
        if vector_labyrinth[node] == 'E':
            end = nodes.index(node)
            end = get_local_position(end, labyrinth_original_shape[0])
        for neighbor in get_neighbors(node, labyrinth.shape, True):
            if vector_labyrinth[neighbor[1]] not in ["-", "|"]:
                edges.append(neighbor)
    return edges, start, end


def find_ice_floors(
    nodes: List[int],
    vector_labyrinth: List[str],
    labyrinth_original_shape: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Find the ice floors in the labyrinth.

    Args:
        nodes (List[int]): list of nodes in the labyrinth.
        vector_labyrinth (List[str]): vector representation of the labyrinth.
        labyrinth_original_shape (Tuple[int, int]): original shape of the labyrinth (e.g., 5x5).

    Returns:
        List[Tuple[int, int]]: ice floors in the labyrinth.
    """
    ice_floors = []
    for node in nodes:
        if vector_labyrinth[node] == 'I':
            floor = nodes.index(node)
            floor = get_local_position(floor, labyrinth_original_shape[0])
            ice_floors.append(tuple(floor))
    return ice_floors


def find_key_and_lock(
    nodes: List[int],
    vector_labyrinth: List[str],
    labyrinth_original_shape: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Find the key and lock in the labyrinth.

    Args:
        nodes (List[int]): list of nodes in the labyrinth.
        vector_labyrinth (List[str]): vector representation of the labyrinth.
        labyrinth_original_shape (Tuple[int, int]): original shape of the labyrinth (e.g., 5x5).

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: key and lock positions.
    """
    key = None
    lock = None
    for node in nodes:
        if vector_labyrinth[node] == 'D':
            lock = nodes.index(node)
            lock = get_local_position(lock, labyrinth_original_shape[0])
        if vector_labyrinth[node] == "K":
            key = nodes.index(node)
            key = get_local_position(key, labyrinth_original_shape[0])
    return key, lock


def invoke_interpreter(path: str) -> Interpreter:
    interpreter = Interpreter()
    interpreter.reset()
    with open(path, "r") as f:
        for line in f:
            interpreter.eval(line)
    return interpreter


def convert_from_file(path: str) -> Tuple[str, Dict[str, bool]]:
    """Convert a labyrinth from a labyrinth file to a str that can be used by the environment.

    Args:
        path (str): path to save the converted labyrinth.
    """
    module = invoke_interpreter(path)
    labyrinth = module.labyrinth
    key_and_lock = module.variables.get("key_and_lock", False)
    icy_floor = module.variables.get("icy_floor", False)

    labyrinth = np.array(labyrinth[::-1])  # labyrinth structure for indexing
    labyrinth_shape = labyrinth.shape
    labyrinth_original_shape = (labyrinth_shape[0] + 1) // 2
    labyrinth_original_shape = (labyrinth_original_shape, labyrinth_original_shape)

    # Get all nodes from the map to find edges.
    vector_labyrinth = labyrinth.reshape((-1))
    nodes = get_nodes(labyrinth_shape)

    # find all edges
    edges, start, end = find_edges_start_and_end(
        nodes,
        vector_labyrinth,
        labyrinth,
        labyrinth_original_shape,
        labyrinth_shape
    )

    # Convert all edges into normal graph values
    converted_edges = []
    for x, y in edges:
        y = convert(x, y, labyrinth_shape[0])
        x = nodes.index(x)
        y = nodes.index(y)
        converted_edges.append((x, y))
    converted_edges.sort()

    # Find ice floors
    if icy_floor:
        ice_floors = find_ice_floors(nodes, vector_labyrinth, labyrinth_original_shape)

    # Find key and lock
    if key_and_lock:
        key, lock = find_key_and_lock(nodes, vector_labyrinth, labyrinth_original_shape)

    save_string = f"{converted_edges};{start};{end}"
    if key_and_lock:
        save_string += f";{key};{lock}"
    if icy_floor:
        save_string += f";{ice_floors}"
    return save_string, module.variables


def create_default_labyrinth(size: Tuple[int, int], path: str) -> None:
    """Create a default labyrinth based on the size.

    Args:
        size (Tuple[int, int]): size of the labyrinth
        path (str): path to save the labyrinth
    """
    w, h = size
    labyrinth = []
    for _ in range(h):
        # Vertical walls
        row = [[" ", "|"] for _ in range(w)]
        row = np.array(row).reshape(-1)[:-1]
        row = list(row)
        labyrinth.append(row)

        # Horizontal walls
        row = [["-", "+"] for _ in range(w)]
        row = np.array(row).reshape(-1)[:-1]
        row = list(row)
        labyrinth.append(row)

    labyrinth = labyrinth[:-1]
    labyrinth[0][w * 2 - 2] = "E"
    labyrinth[h * 2 - 2][0] = "S"

    if ".py" not in path:
        path += ".labyrinth"
    write_into_file(labyrinth, path)


def create_file_from_environment(environment: Env, path: str) -> None:
    """Create a labyrinth file from an environment.

    Args:
        environment (Env): environment to create the labyrinth from
        path (str): path to save the labyrinth
    """
    structure = environment.save("").split(";")
    pathways = ast.literal_eval(structure[0])
    start = ast.literal_eval(structure[1])
    end = ast.literal_eval(structure[2])

    w, h = environment.shape
    labyrinth = []
    for i in range(h):
        # Vertical walls
        row = []
        for j in range(w):
            if j == w - 1:
                row.append([" ", "|"])
            else:
                current = environment.get_global_position((i, j))
                next = environment.get_global_position((i, j + 1))
                if (current, next) in pathways or (next, current) in pathways:
                    row.append([" ", " "])
                else:
                    row.append([" ", "|"])
        row = np.array(row).reshape(-1)[:-1]
        row = list(row)
        labyrinth.append(row)

        # Horizontal walls
        if i < h - 1:
            row = []
            for j in range(w):
                current = environment.get_global_position((i, j))
                next = environment.get_global_position((i + 1, j))
                if (current, next) in pathways or (next, current) in pathways:
                    row.append([" ", "+"])
                else:
                    row.append(["-", "+"])
            row = np.array(row).reshape(-1)[:-1]
            row = list(row)
            labyrinth.append(row)

    labyrinth[start[0] * 2][start[1] * 2] = 'S'
    labyrinth[end[0] * 2][end[1] * 2] = 'E'

    if environment.icy_floor:
        ice_floors = ast.literal_eval(structure[-1])
        for ice_floor in ice_floors:
            labyrinth[ice_floor[0] * 2][ice_floor[1] * 2] = 'I'

    if environment.key_and_door:
        key = ast.literal_eval(structure[-2])
        door = ast.literal_eval(structure[-1])
        labyrinth[key[0] * 2][key[1] * 2] = 'K'
        labyrinth[door[0] * 2][door[1] * 2] = 'D'

    labyrinth = labyrinth[::-1]

    variables = {
        "key_and_lock": environment.key_and_door,
        "icy_floor": environment.icy_floor,
        "occlusion": environment.occlusion
    }
    write_into_file(labyrinth, path, variables)


def write_into_file(
    labyrinth: List[List[str]],
    path: str,
    variables: Dict[str, bool] = None
) -> None:
    """Write the labyrinth into a file.

    Args:
        labyrinth (List[List[str]]): labyrinth to be written
        path (str): path to save the labyrinth
    """
    key_and_lock = False if variables is None else variables["key_and_lock"]
    icy_floor = False if variables is None else variables["icy_floor"]
    occlusion = False if variables is None else variables["occlusion"]

    with open(path, "w", encoding="utf-8") as _file:
        _file.write('"""\nThis file was created automatically.')
        _file.write('\nFor more instructions read the README.md\n"""\n')
        _file.write(f"key_and_lock: {key_and_lock}\n")
        _file.write(f"icy_floor: {icy_floor}\n")
        _file.write(f"occlusion: {occlusion}\n\n")
        _file.write("labyrinth:\n")
        _file.write("-" * ((len(labyrinth[0]) + 2) * 2 - 1) + "\n")
        for row in labyrinth:
            _file.write("| ")
            for token in row:
                _file.write(f"{token} ")
            _file.write("|\n")
        _file.write("-" * ((len(labyrinth[0]) + 2) * 2 - 1) + "\n")
        _file.write("end")
