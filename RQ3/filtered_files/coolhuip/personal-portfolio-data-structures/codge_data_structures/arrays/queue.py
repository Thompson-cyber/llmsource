# Hey, Copilot! I want you to give me a data structure assignment where I have
# to implement a class (and its standard methods) with custom code. Thanks!

# Path: codge_data_structures\arrays\queue.py
# Compare this snippet from codge_data_structures\arrays\stack.py:
"""
Day 2: Queue

Docstring Generated by Copilot:
===============================
Queues are another common data structure. You should know how to implement
one. The basic idea is that you can add and remove items from the "front" of
the queue in O(1) time. You can think of it like a line at a grocery store.
If you want to add a person to the line, you just place them at the back. If
you want to remove a person from the line, you always remove them from the
front. First person added is the first person that gets removed. This is
called FIFO (First In First Out).

I'm going to need you to do some work for me. I need you to write a class for
a queue. It should have the methods enqueue, dequeue, and size. FIFO (first
in first out) means that the first element added to the queue should be the
first one removed. So if you enqueue 1, and then enqueue 2, and then enqueue
3, you'll need to dequeue 1 first, then 2, then 3. Make sure you return the
value that is being dequeued.

Example usage:
q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.size() # should return 3
q.dequeue() # should return 1
q.dequeue() # should return 2
q.dequeue() # should return 3
q.size() # should return 0
"""
from __future__ import annotations
from typing import Any, Union, Optional






if __name__ == '__main__':
    import doctest
    print()
    doctest.testmod()
    print()
