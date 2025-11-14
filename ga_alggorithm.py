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


def make_onemax(dim: int) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        return float(np.sum(x))

    return GAProblem(
        name=f"OneMax ({dim} bits)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )


# ⭐ Required Custom Problem ⭐
def make_custom_50_of_80() -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        return 80 - abs(ones - 50)  # max at ones = 50

    return GAProblem(
        name="Custom 50-of-80 Problem (max when ones = 50)",
        chromosome_type="bit",
        dim=80,
        bounds=None,
        fitness_fn=fitness,
    )


def make_sphere(dim: int, lo: float, hi: float) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        return -float(np.sum(np.square(x)))
    return GAProblem("Sphere", "real", dim, (lo, hi), fitness)


def make_rastrigin(dim: int, lo: float, hi: float) -> GAProblem:
    def rastrigin(x: np.ndarray) -> float:
        A = 10
        return A * len(x) + np.sum(x * x - A * np.cos(2 * np.pi * x))

    def fitness(x):
        return -rastrigin(x)

    return GAProblem("Rastrigin", "real", dim, (lo, hi), fitness)


# -------------------- GA Operators --------------------
def init_population(problem, pop_size, rng):
    if problem.chromosome_type == "bit":
        return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)
    else:
        lo, hi = problem.bounds
        return rng.uniform(lo, hi, size=(pop_size, problem.dim))


def tournament_selection(fitness, k, rng):
    idxs = rng.integers(0, len(fitness), size=k)
    return int(idxs[np.argmax(fitness[idxs])])


def one_point_crossover(a, b, rng):
    if len(a) <= 1:
        return a.copy(), b.copy()
    point = rng.integers(1, len(a))
    return np.concatenate([a[:point], b[point:]]), np.concatenate([b[:point], a[point:]])


def arithmetic_crossover(a, b, rng):
    alpha = rng.random(a.shape)
    return alpha * a + (1 - alpha) * b, alpha * b + (1 - alpha) * a


def bit_mutation(x, mut_rate, rng):
    mask = rng.random(len(x)) < mut_rate
    y = x.copy()
    y[mask] ^= 1
    return y


def gaussian_mutation(x, mut_rate, sigma, rng, bounds):
    y = x.copy()
    mask = rng.random(len(x)) < mut_rate
    noise = rng.normal(0, sigma, len(x))
    y[mask] += noise[mask]
    lo, hi = bounds
    return np.clip(y, lo, hi)


def evaluate(pop, problem):
    return np.array([problem.fitness_fn(ind) for ind in pop])


# -------------------- GA Run --------------------
def run_ga(
    problem,
    pop_size,
    generations,
    crossover_rate,
    mutation_rate,
    tournament_k,
    elitism,
    real_sigma,
    seed,
    stream_live=True,
):
    rng = np.random.default_rng(seed)

    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    chart_area = st.empty()
    best_area = st.empty()

    history_best, history_avg, history_worst = [], [], []

    for gen in range(generations):
        best = fit.max()
        avg = fit.mean()
        worst = fit.min()

        history_best.append(best)
        history_avg.append(avg)
        history_worst.append(worst)

        if stream_live:
            df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
            chart_area.line_chart(df)
            best_area.markdown(f"Generation {gen+1}/{generations} — Best: **{best:.4f}**")

        # Elitism
        E = elitism
        elite_idx = np.argpartition(fit, -E)[-E:]
        elites = pop[elite_idx]

        # New generation
        new_pop = []
        while len(new_pop) < pop_size - E:
            p1 = pop[tournament_selection(fit, tournament_k, rng)]
            p2 = pop[tournament_selection(fit, tournament_k, rng)]

            # Crossover
            if rng.random() < crossover_rate:
                if problem.chromosome_type == "bit":
                    c1, c2 = one_point_crossover(p1, p2, rng)
                else:
                    c1, c2 = arithmetic_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            if problem.chromosome_type == "bit":
                c1 = bit_mutation(c1, mutation_rate, rng)
                c2 = bit_mutation(c2, mutation_rate, rng)
            else:
                c1 = gaussian_mutation(c1, mutation_rate, real_sigma, rng, problem.bounds)
                c2 = gaussian_mutation(c2, mutation_rate, real_sigma, rng, problem.bounds)

            new_pop.append(c1)
            if len(new_pop) < pop_size - E:
                new_pop.append(c2)

        # Add elites
        pop = np.vstack([new_pop, elites])
        fit = evaluate(pop, problem)

    return pop, fit, history_best, history_avg, history_worst

