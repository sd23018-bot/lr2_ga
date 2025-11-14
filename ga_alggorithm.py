import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Problem Definitions --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  # 'bit' or 'real'
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


# ===== CUSTOM PROBLEM (assignment requirement) =====
def make_custom(dim: int = 80) -> GAProblem:
    # Fitness reaches maximum (80) when number of ones = 50
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        return float(80 - abs(ones - 50))

    return GAProblem(
        name="Custom 80-bit Pattern",
        chromosome_type="bit",
        dim=80,
        bounds=None,
        fitness_fn=fitness,
    )


# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)

def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)

def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y

def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


def run_ga(
    problem: GAProblem,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_k: int,
    elitis
