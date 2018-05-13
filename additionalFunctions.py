# coding=utf-8
def calculate_error(expected, real):
    return abs((real - expected) / expected)

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
