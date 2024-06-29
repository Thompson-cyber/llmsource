# quick.py
"""
A quick-sort visualizer (using matplot).

This code was generate by ChatGPT and then
(just) polished/refactored by me.

The main things that were/are missing are ...

* no tests
* no logging
* no (good) design (modularisation)
* no taste (not elegant/beautiful)
* not working (plt.clf() was missing)

The generated code is not production quality.
"""

import matplotlib.pyplot as plt
import random
import util


def quicksort(arr, left, right):
    """Sort the array and display the sorting."""
    if left >= right:
        return
    pivot = arr[random.randint(left, right)]
    i, j = left, right
    while i <= j:
        while arr[i] < pivot:
            i += 1
        while arr[j] > pivot:
            j -= 1
        if i <= j:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j -= 1
        draw(arr)
    quicksort(arr, left, j)
    quicksort(arr, i, right)


def draw(arr):
    """Draw the given array."""
    plt.clf()
    plt.bar(range(len(arr)), arr)
    plt.show(block=False)
    plt.pause(0.1)


# Example usage
arr = util.randlist(100, 100, 1)
quicksort(arr, 0, len(arr) - 1)
print(arr)
