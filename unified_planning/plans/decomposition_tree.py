from typing import List, Dict, Set, Tuple, Union, cast

from collections import OrderedDict
from enum import Enum, auto

import unified_planning as up

class DecompositionTreeNodeKind(Enum):
    """
    Enum referring to the possible kinds of `DecompositionTreeNode`.
    """

    TASK_NODE = auto()
    ACTION_NODE = auto()



class DecompositionTreeNode():
    def __init__(
        self,
        id: int, # Integer ID given by the planner
        type: DecompositionTreeNodeKind,
        name: str,
        depth: int,
        method: str = None,
        children: List[int] = [],
        objects: List["up.model.Object"] = [],
        unif: List["up.model.Parameter"] = []
    ):
        self._id = id
        self._name = name
        self._method = method
        self._type = type

        self._children = children

        self._depth = depth
    
        self._unifications = OrderedDict({x: y for x,y in zip(unif, objects)})
        self._objects = objects

    def __repr__(self) -> str:
        s = []
        s.append(f"{self._id}: ")
        s.append(
            "task " if self._type == DecompositionTreeNodeKind.TASK_NODE
                        else "action "
        )
        s.append(f"{self._name}\n")

        if self._type == DecompositionTreeNodeKind.TASK_NODE:
            s.append(f" - Method: {self._method}\n")

        s.append(f" - Depth: {self._depth}\n")
        s.append(f" - Objects: {' '.join([str(o) for o in self._objects])}\n")
        # TODO: Unifications

        if self._type == DecompositionTreeNodeKind.TASK_NODE:
            s.append(f" - Children: {' '.join([str(c) for c in self._children])}\n")

        return "".join(s)

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def task(self) -> str: # TODO: Return Task() ???
        assert self._type == DecompositionTreeNodeKind.TASK_NODE
        return self._name

    @property
    def method(self) -> str: # TODO: Return Method() ???
        assert self._type == DecompositionTreeNodeKind.TASK_NODE
        return self._method

    @property
    def type(self) -> DecompositionTreeNodeKind:
        return self._type

    def is_task(self) -> bool:
        return self._type == DecompositionTreeNodeKind.TASK_NODE

    def is_action(self) -> bool:
        return self._type == DecompositionTreeNodeKind.ACTION_NODE

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def children(self) -> List[int]:
        return self._children

    @property
    def objects(self) -> List[str]:
        return self._objects

    @property
    def unifications(self) -> OrderedDict["up.model.Parameter", "up.model.Object"]:
        """Get dictionary of unifications.
        An unification indicates what parameters from the domain are transformed
        into which objects to obtain a plan"""
        return self._unifications


    def is_task(self) -> bool:
        return self._type == DecompositionTreeNodeKind.TASK_NODE

    def is_action(self) -> bool:
        return self._type == DecompositionTreeNodeKind.ACTION_NODE



class DecompositionTree():
    """Stores de decomposition tree of a resulting plan"""

    def __init__(
        self,
        problem: "up.model.Problem",
        tree: str,   # Upper part of the DT, non-leaves
        plan: str, # Leaves in the DT
    ) -> None:
        self._tree_str = plan + '\n' + tree

        # Tree: Node X is this DecompositionTreeNode
        self._index: Dict[int, List[int]] = OrderedDict()
        self._tree: Dict[int, List[DecompositionTreeNode]] = OrderedDict()

        # TODO: Maybe change _ to - before, undo hpdl_writer name convention
        dt = tree.splitlines()

        # Root, top-most tasks indexs, appears in the first line
        # Format: root 0 1 2 3
        dt[0] = dt[0].removeprefix('root ')
        dt[0] = dt[0].split(' ')
        self._root = [int(x) for x in dt[0]]

        # To store depth of each node
        self._depth = 0 # Maximum depth
        self._depths = {} # depth: List[ids]
        for t in self._root:
            self._depths[t] = 0

        # The rest are nodes
        for node_str in dt[1:]:
            node_str = node_str.strip() # Remove trailing spaces

            # Format:
            # 0 get_data objective1 high_res -> m_get_image_data_0 3 4 5 6
            line = node_str.split(' -> ')
            task = line[0].split(' ')
            method = line[1].split(' ')

            id = int(task[0])
            name = task[1]
            objects = [problem.object(x) for x in task[2:]]
            child_ids = [int(x) for x in method[1:]]
            method_name = method[0]

            depth = self._depths[id]

            # Access UPF task and get unifications
            if problem.has_task(name):
                unif = problem.get_task(name).parameters
            else:
                raise Exception("Couldn't find task {name}")

            # Create DecompositionTreeNode
            dt_node = DecompositionTreeNode(
                id=id,
                type=DecompositionTreeNodeKind.TASK_NODE,
                name=name,
                depth=depth,
                method=method_name,
                children=child_ids,
                objects=objects,
                unif=unif
            )

            # Update depth of children, one more than current node
            depth += 1
            for c in child_ids:
                self._depths[c] = depth

            # Store node
            self._index[id] = child_ids
            self._tree[id] = dt_node


        # Add leaves (primitives) to the structure
        self._leaves = []

        for node_str in plan.splitlines():
            node_str = node_str.strip() # Remove trailing spaces

            # Format: 
            # 27 navigate rover0 waypoint3 waypoint1
            task = node_str.split(' ')

            id = int(task[0])
            name = task[1] + '_primitive'
            objects = [problem.object(x) for x in task[2:]]
            depth = self._depths[id]

            # Access UPF task and get unifications
            if problem.has_action(name):
                unif = problem.action(name).parameters
            else:
                raise Exception("Couldn't find task {name}")

            # Create DecompositionTreeNode
            dt_node = DecompositionTreeNode(
                id=id,
                type=DecompositionTreeNodeKind.ACTION_NODE,
                name=name,
                depth=depth,
                method=None,
                children=[],
                objects=objects,
                unif=unif
            )

            # Store node
            self._index[id] = []
            self._tree[id] = dt_node

            self._leaves.append(id)
    
            # Get maximum depth
            self._depth = depth if depth > self._depth else self._depth

        
        # Get dict {depth: List[node_id]}
        self._nodes_at_depth = {d: [] for d in range(0,self.depth+1)}

        for k, v in self._depths.items():
            self._nodes_at_depth[v].append(k)



    def __repr__(self) -> str:
        """Using the IPC 2020 representation format"""
        s = []
        return "\n".join(str(node) for id, node in self._tree.items())


    @property
    def root(self) -> List[int]:
        """Return indexs of root nodes (task-goal)"""
        return self._root

    @property
    def root_nodes(self) -> List[DecompositionTreeNode]:
        """Return list of root nodes (task-goal) as DecompositionTreeNode"""
        return [self._tree[id] for id in self._root]
    
    @property
    def leaves(self) -> List[int]:
        """Return indexs of leaves nodes (actions)"""
        return self._leaves
    
    @property
    def leaves_nodes(self) -> List[DecompositionTreeNode]:
        """Return list of leaves nodes (actions) as DecompositionTreeNode"""
        return [self._tree[id] for id in self._leaves]

    @property
    def depth(self) -> int:
        """Return maximum depth of the tree"""
        return self._depth

    def num_actions(self) -> int:
        """Return number of actions (non-tasks) on the tree"""
        return len(self._leaves)

    def num_tasks(self) -> int:
        """Return number of tasks (non-actions) on the tree"""
        return len(self._index) - len(self._leaves)

    def nodes_at_depth(self, depth: int) -> List[int]:
        """Return IDs of nodes at given depth"""
        return self._nodes_at_depth[depth]

    # TODO: Should also accept list?
    def node(self, id: int) -> DecompositionTreeNode:
        """Return node @id as a DecompositionTreeNode"""
        return self._tree[id]

    def children(self, id: int) -> List[int]:
        """Return indexs of children of node @id"""
        assert id in self._index
        return self._index[id]

    # TODO
    def node_id(self, node: DecompositionTreeNode) -> int:
        """Returns ID of @node"""
        return node.id

    # TODO: Upper part of tree


    def plot(self, names=True, objects=True) -> str:
        """Plot tree and return it as string.

        names: Print names of each node
        objects: Print objects of each node
        """
        s = []
        
        for id in self._root:
            s.extend(self._plot_node(id, names, objects))

        return '\n'.join(s)
        

    def _plot_node(self, id, names=True, objects=True) -> List[str]:
        """Plots one node and recursively plots children"""
        s = []

        # Number of tabs at the start of the string
        tabs = self._depths[id]

        # Print
        string = f"{'  ' * tabs} |-{id}"

        if names:
            string += ' ' + self.node(id).name

        if objects:
            string += '(' + ', '.join(
                            [str(o) + ' - ' + str(o.type) for o in self.node(id).objects]
                        ) + ')'

        s.append(string)
        
        # Recursion
        for c in self.children(id):
            s.extend(self._plot_node(c, names, objects))

        return s

    
    # FIXME: Won't show '_primitive' in actions
    def print_as_ipc(self) -> str:
        """Print decomposition tree following the IPC2020 format"""
        return self._tree_str