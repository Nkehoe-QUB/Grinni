import os


def Open(*args, **kwargs):
    from _Process.py import Process
    return Process(*args, **kwargs)