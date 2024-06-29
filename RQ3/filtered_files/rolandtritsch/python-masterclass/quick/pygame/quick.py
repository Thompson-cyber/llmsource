# quick.py
"""
A quick-sort visualizer (using pygame).

This code was generate by ChatGPT and then
(just) polished/refactored by me.
"""

from constants import Constants
import pygame
import random
import util


def draw_array(screen, arr, left_index, right_index, current_index):
    """Draw the array on the screen."""
    screen.fill(Constants.WHITE)
    bar_width = Constants.WIDTH // len(arr)
    for i, value in enumerate(arr):
        color = Constants.BLACK
        if i == current_index:
            color = Constants.GREEN
        elif i < left_index or i > right_index:
            color = Constants.RED
        pygame.draw.rect(
            screen,
            color,
            (i * bar_width, Constants.HEIGHT - value, bar_width, value)
        )
    pygame.display.update()


def quicksort(screen, arr, left, right):
    """Sort the array (and draw it, while sorting it)."""
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
            draw_array(screen, arr, left, right, j)
            i += 1
            j -= 1
        else:
            draw_array(screen, arr, left, right, i)
        pygame.time.delay(1000)
    quicksort(screen, arr, left, j)
    quicksort(screen, arr, i, right)


# Initialize pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode(
    (Constants.WIDTH, Constants.HEIGHT)
)
pygame.display.set_caption(
    Constants.CAPTION
)

# Example usage
arr = util.randlist(7, 10, 50)
quicksort(screen, arr, 0, len(arr) - 1)

# Quit pygame
pygame.time.delay(3000)
pygame.quit()
