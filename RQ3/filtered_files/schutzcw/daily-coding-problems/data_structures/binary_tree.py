#!/usr/bin/env python3

# TODO: mypy
# TODO: pytest
# TODO: Add a remove function

# Questions: 
# TODO: How to re-balance a binary tree?

from enum import auto, Enum, unique
from queue import Queue
from typing import Generic, List, Optional, TypeVar, Union


import logging
import random

logging.basicConfig(level=logging.DEBUG)


@unique
class LRType(Enum):
    ROOT = auto()
    LEFT = auto()
    RIGHT = auto()


T = TypeVar("T")


class Node(Generic[T]):
    """ Single node in a binary tree. """

    value: int
    left: Optional["Node[T]"]
    right: Optional["Node[T]"]
    parent: Optional["Node[T]"]

    def __init__(self,
                 val: int,
                 left: Optional["Node[T]"] = None,
                 right: Optional["Node[T]"] = None,
                 parent: Optional["Node[T]"] = None) -> None:
        """ """
        self.value = val
        self.left = left
        self.right = right
        self.parent = parent

    def is_leaf(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return True if (self.right is None) and (self.left is None) else False

    def __repr__(self) -> str:
        """ """
        return f"({self.value})"


class BinaryTree(Generic[T]):
    """ Binary Tree """

    root = Optional[Node[T]]
    node_count: int
    depth: int


    def __init__(self, 
                 root: Optional[Node[T]] = None,
                 node_count: int = 0,
                 depth: int = 0):
        """ """
        self.root = root
        self.node_count = node_count
        self.depth = depth


    def get_leaves(self) -> List[Node[T]]:
        """

        Returns:
            List[Node[T]]: _description_
        """
        
        list_ = []
        if self.root is None:
            return list_
        
        queue = Queue()
        queue.put_nowait(self.root)
        while not queue.empty():
            node = queue.get()
            if node.left is not None:
                queue.put_nowait(node.left)
            if node.right is not None:
                queue.put_nowait(node.right)
            if (node.left is None) and (node.right is None):
                list_.append(node)
        return list_        
        
        
    def add(self, node: Union[Node[T], T]) -> bool:
        """Add a new value to the tree

        Args:
            val (T): The value to add to the tree

        Returns:
            bool: True if the value was added to the tree, otherwise false.
        """

        if not isinstance(node, Node):
            node = Node(node)

        # Empty tree, add new node as root
        if self.node_count == 0:
            self.root = node
            self.node_count = 1
            self.depth = 1
            return True

        return self._add(node, self.root)


    def is_empty(self) -> bool:
        """Determine if the tree is empty.

        Returns:
            bool: True if the tree is empty. Otherwise False.
        """
        return self.root is None


    def _add(self, new_node: Node, node: Node, depth: int = 2) -> bool:
        """Internal function logic for adding a value to the tree.

        Args:
            new_node (Node): The new node that we're trying to add
            node (Node): The current node that we're comparing to.
            depth (int): The current depth.

        Returns:
            bool: True if the value was added to the tree, otherwise false.
        """
        

        # Node is already in tree, don't add
        if new_node.value == node.value:
            return False

        if new_node.value < node.value:
            if node.left is None:
                new_node.parent = node
                node.left = new_node
                self._increment(depth)
                return True
            else:
                return self._add(new_node, node.left, depth + 1)
        
        if node.right is None:
            new_node.parent = node
            node.right = new_node
            self._increment(depth)  
            return True
        
        return self._add(new_node, node.right, depth + 1)


    def _increment(self, depth: int) -> None:
        """Increase node count and possibly depth.
        
        Arguments:
            depth: The depth at which we added a node.
        """
        self.node_count +=1
        if depth > self.depth:
            self.depth += 1


    def printTree(self, node: Optional[Node] = None, extra: bool = True) -> None:
        """Print the tree and possibly some additional stats.

        Args:
            node (Node | None): The Node to root the print at. If None, use the tree's root Node.
            extra (bool, optional): Whether to add additional information to the print, such as
            `node count` and `tree depth`. Defaults to True.
        """
        if node is None:
            node = self.root

        self._printTree(node)
        if extra:
            print(f"Node Count: {self.node_count}")
            print(f"Depth: {self.depth}")


    def _printTree(self, node: Node, prefix: str = "", lrtype = LRType.ROOT) -> None:
        """ Print function generated by ChatGPT. Prints the tree horizontally, where the right
        sub-tree is printed at the top and the left-subtree at the bottom.

        Args:
            node (Node): The starting node to print (usually the root node)
            prefix (str, optional): Not to be specified by the user. Helps this recursive function
                track status for printing. Defaults to "".
            lrtype (bool, optional): Not to be specified by the user. Helps this recursive function
                track status for printing. Defaults to LRType.ROOT.
        """

        if node is None:
            return

        # always print right branch first
        if node.right is not None:
            if lrtype == LRType.ROOT:
                add_prefix = ""
            elif lrtype == LRType.RIGHT:
                add_prefix = "     "
            else:
                add_prefix = "│    "             
            self._printTree(node.right, prefix + add_prefix, LRType.RIGHT)
        
        # when right subtree is done, print current node
        if lrtype == LRType.ROOT:
            add_prefix = ""
        elif lrtype == LRType.RIGHT:
            add_prefix = "┌─── "
        else:
            add_prefix = "└─── "
        print(prefix + add_prefix + str(node.value))

        # when right subtree and current node are printed, start printing left subtree
        if node.left is not None:
            if lrtype == LRType.ROOT:
                add_prefix = ""
            elif lrtype == LRType.RIGHT:
                add_prefix = "│    "
            else:
                add_prefix = "     "               
            self._printTree(node.left, prefix + add_prefix, LRType.LEFT)


    def find(self, search_node: Union[Node[T], T]) -> Optional[Node[T]]:
        """_summary_

        Args:
            node (Union[Node[T], T]): _description_

        Returns:
            Optional[Node[T]]: _description_
        """
        if not isinstance(search_node, Node):
            search_node = Node(search_node)

        nodes = [self.root]
        while(len(nodes) > 0):
            new_nodes = []
            for _ in nodes:
                node = nodes.pop()
                if node.value == search_node.value:
                    return node

                if node.left is not None:
                    new_nodes.append(node.left)
                if node.right is not None:
                    new_nodes.append(node.right)
            nodes = nodes + new_nodes
        return None


    def remove(self, node: Node) -> bool:
        raise NotImplementedError()


    def search(self, node: Union[Node[T], T]) -> bool:
        """Search the tree to see if the value currently exists in the tree.

        Args:
            node (Union[Node[T], T]): This can be a value or a Node containing a value.

        Returns:
            bool: True if the value exists in the tree. Otherwise False.
        """
        if self.root is None:
            return False

        if not isinstance(node, Node):
            node: Node[T] = Node(node)
        return self._search(node, self.root)


    def _search(self, search_node: Node[T], current_node: Optional[Node[T]]) -> bool:
        """Search the tree to see if the value currently exists in the tree.

        Args:
            search_node (Node): The Node that we're searching for.
            current_node (Node): The current Node that we're inspecting.

        Returns:
            bool: True if the value exists in the tree. Otherwise False.
        """

        if current_node is None:
            return False

        if search_node.value == current_node.value:
            return True

        if search_node.value < current_node.value:
            return self._search(search_node, current_node.left)

        return self._search(search_node, current_node.right)


    def list(self) -> List[T]:
        """Return all node values as a list of elements.

        Returns:
            List[T]: A list containing all the nodes values
        """
        values = []
        nodes = [self.root]
        while len(nodes) > 0:
            node = nodes.pop()
            values.append(node.value)
            if node.left is not None:
                nodes.append(node.left)
            if node.right is not None:
                nodes.append(node.right)
        values.sort()
        return values


    def random_value(self) -> Optional[T]:
        """Returns a random value from within the tree

        Returns:
            T: A random node value already in the tree. None if no values exist.
        """
        if self.root is None:
            return None

        vals = self.list()
        return random.choice(vals)


    # TODO: This tree is not self balancing, so we can't return log2(node_count)...
    # def depth() -> int:
    #     """ return: The depth of the tree. """
    #     if node_count < 2:
    #         return node_count
        
    #     if node_count < 3:
    #         return 1
        
    #     return 
    

def print_separator():
    """ """
    print("*"*75)


def create_tree(n: int, min_val: int = 0, max_val: int = 100, seed: int = 0) -> BinaryTree[int]:
    """Create a binary tree with n random integers

    Args:
        n (int): The number of random integers to place in the tree
        min_val (int, optional): The lowest possible integer value in the tree. Defaults to 0.
        max_val (int, optional): The highest possible integer value in the tree. Defaults to 100.
        seed (int, optional): Seed for random number generation. If -1, then use random seeding. 
            Defaults to -1.

    Raises:
        RuntimeError: Error raised if the conditions are impossible

    Returns:
        BinaryTree[int]: The created tree
    """
    
    if (max_val - min_val) < n:
        raise RuntimeError("max_val - min_val < n")
    
    tree = BinaryTree()
    for _ in range(n):
        already_found = True
        while already_found:
            val = random.randint(min_val, max_val)
            if not tree.search(val):
                already_found = False
                tree.add(val)
                if val == min_val:
                    min_val = min_val + 1
                if val == max_val:
                    max_val = max_val - 1
    return tree


def main():
    """ """
    SEED = -1
    import time
    if SEED == -1:
        random.seed(time.time())
    else:
        random.seed(SEED)

    tree = create_tree(40)
    tree.printTree(tree.root)

    # print(f"search(7): {tree.search(7)}")
    # print(f"search(6): {tree.search(6)}")
    # node = tree.find(1)
    # if node is not None:
    #     print(f"parent: {node.parent}")
    #     print(f"left: {node.left}")
    #     print(f"right: {node.right}")

if __name__ == "__main__":
    main()