from enum import Enum


class DNSRCode(Enum):
    "Generated with ChatGPT."
    UNKNOWN = None
    NOERROR = 0  # No Error
    FORMERR = 1  # Format Error
    SERVFAIL = 2  # Server Failure
    NXDOMAIN = 3  # Non-Existent Domain
    NOTIMP = 4  # Not Implemented
    REFUSED = 5  # Query Refused
    YXDOMAIN = 6  # Name Exists when it should not
    YXRRSET = 7  # RR Set Exists when it should not
    NXRRSET = 8  # RR Set that should exist does not
    NOTAUTH = 9  # Server Not Authoritative for zone
    NOTZONE = 10  # Name not contained in zone

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN


import tensorflow as tf
from tensorflow.keras import layers


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def print_paths_to_leaves(root):
    def dfs(node, path):
        ......
        # Pass a copy of the current path
        dfs(node.left, path.copy())
        dfs(node.right, path.copy())

    if not root: return
    dfs(root, [])
# Define a simple feedforward neural network model
def create_model(input_shape, num_classes):
    # Creating a Sequential model
    # Adding a dense layer with ReLU activation
    # Adding output layer with softmax activation for classification
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu',
                     input_shape = input_shape),
        layers.Dense(num_classes,activation='softmax')
    ])
    return model
