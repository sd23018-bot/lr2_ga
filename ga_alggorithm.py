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
    chromosome_type: str  # 'bit'
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


# Custom fitness: maximum when ones = 50, max fitness = 80
def make_custom_50_of_80():
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        return 80 - abs(ones - 50)   # MAX = 80 at ones=50

    return GAProblem(
        name="Custom bit problem (ones=50 gives max fitness)",
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
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator):
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


# -------------------- Run GA --------------------
def run_ga(problem, pop_size, generations, crossover_rate, mutation_rate,
           tournament_k, elitism, seed, stream_live):

    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    # containers
    chart_area = st.empty()
    best_area = st.empty()

    history_best, history_avg, history_worst = [], [], []

    for gen in range(generations):

        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])

        history_best.append(best_fit)
        history_avg.append(float(np.mean(fit)))
        history_worst.append(float(np.min(fit)))

        # Live update
        if stream_live:
            df = pd.DataFrame({"Best": history_best, "Avg": history_avg, "Worst": history_worst})
            chart_area.line_chart(df)
            best_area.markdown(f"Generation {gen+1}/{generations} — **Best fitness = {best_fit:.2f}**")

        # Elitism
        E = elitism
        elite_idx = np.argpartition(fit, -E)[-E:]
        elites = pop[elite_idx].copy()

        # Next generation
        next_pop = []

        while len(next_pop) < pop_size - E:
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        pop = np.vstack([next_pop, elites])
        fit = evaluate(pop, problem)

    # return results
    best_idx = int(np.argmax(fit))
    return {
        "best": pop[best_idx],
        "best_fitness": fit[best_idx],
        "final_population": pop,
        "final_fitness": fit,
        "history": pd.DataFrame({"Best": history_best, "Avg": history_avg, "Worst": histo
