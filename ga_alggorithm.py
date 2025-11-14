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
        return float(np.sum(x))  # maximize number of ones

    return GAProblem(
        name=f"OneMax ({dim} bits)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )


# ⭐ NEW REQUIRED PROBLEM ⭐
def make_custom_50_of_80() -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        return 80 - abs(ones - 50)  # maximum = 80 when ones = 50

    return GAProblem(
        name="Custom 50-of-80 Problem (max when ones = 50)",
        chromosome_type="bit",
        dim=80,     # fixed chromosome size
        bounds=None,
        fitness_fn=fitness,
    )


def make_sphere(dim: int, lo: float, hi: float) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        return -float(np.sum(np.square(x)))

    return GAProblem(
        name=f"Sphere {dim}D (maximize -||x||^2)",
        chromosome_type="real",
        dim=dim,
        bounds=(lo, hi),
        fitness_fn=fitness,
    )


def make_rastrigin(dim: int, lo: float, hi: float) -> GAProblem:
    def rastrigin(x: np.ndarray) -> float:
        A = 10.0
        return float(A * x.size + np.sum(x * x - A * np.cos(2 * np.pi * x)))

    def fitness(x: np.ndarray) -> float:
        return -rastrigin(x)

    return GAProblem(
        name=f"Rastrigin {dim}D (maximize -f)",
        chromosome_type="real",
        dim=dim,
        bounds=(lo, hi),
        fitness_fn=fitness,
    )


# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    if problem.chromosome_type == "bit":
        return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)
    else:
        lo, hi = problem.bounds
        return rng.uniform(lo, hi, size=(pop_size, problem.dim))


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    return int(idxs[np.argmax(fitness[idxs])])


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def arithmetic_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    alpha = rng.random(a.shape)
    c1 = alpha * a + (1 - alpha) * b
    c2 = alpha * b + (1 - alpha) * a
    return c1, c2


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator):
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def gaussian_mutation(x: np.ndarray, mut_rate: float, sigma: float, rng: np.random.Generator, bounds):
    y = x.copy()
    mask = rng.random(x.shape) < mut_rate
    noise = rng.normal(0.0, sigma, size=x.shape)
    y[mask] += noise[mask]
    lo, hi = bounds
    np.clip(y, lo, hi, out=y)
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
    elitism: int,
    real_sigma: float,
    seed: int | None,
    stream_live: bool = True,
):
    rng = np.random.default_rng(seed)

    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    chart_area = st.empty()
    best_area = st.empty()

    history_best = []
    history_avg = []
    history_worst = []

    for gen in range(generations):

        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))

        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        if stream_live:
            df = pd.DataFrame(
                {"Best": history_best, "Average": history_avg, "Worst": history_worst}
            )
            chart_area.line_chart(df)
            best_area.markdown(
                f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.6f}**"
            )

        # Elitism
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy()
        elites_fit = fit[elite_idx].copy()

        # Create next generation
        next_pop = []

        while len(next_pop) < pop_size - E:

            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

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

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        pop = np.vstack([np.array(next_pop), elites])
        fit = evaluate(pop, problem)

    # Final results
    best_idx = np.argmax(fit)
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    return {
        "best": best,
        "best_fitness": best_fit,
        "history": df,
        "final_population": pop,
        "final_fitness": fit,
    }

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm (GA)")

with st.sidebar:
    st.header("Problem")

    problem_type = st.selectbox(
        "Type",
        ["Custom 50-of-80 (bits)", "OneMax (bits)", "Sphere (real)", "Rastrigin (real)"]
    )

    if problem_type == "Custom 50-of-80 (bits)":
        problem = make_custom_50_of_80()

    elif problem_type == "OneMax (bits)":
        problem = make_onemax(80)  # fixed length

    else:
        dim = st.number_input("Dimension", min_value=2, max_value=256, value=10)
        lo = st.number_input("Lower bound", value=-5.12)
        hi = st.number_input("Upper bound", value=5.12)

        if problem_type == "Sphere (real)":
            problem = make_sphere(int(dim), float(lo), float(hi))
        else:
            problem = make_rastrigin(int(dim), float(lo), float(hi))

    st.header("GA Parameters")

    # ⭐ Fixed required parameters ⭐
    pop_size = 300
    generations = 50

    st.write("Population size: **300 (fixed)**")
    st.write("Generations: **50 (fixed)**")

    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.9)
    mutation_rate = st.slider("Mutation rate (per gene)", 0.0, 1.0, 0.01)
    tournament_k = st.slider("Tournament size", 2, 10, 3)
    elitism = st.slider("Elites per generation", 0, 100, 2)
    real_sigma = st.number_input("Real-valued mutation sigma", value=0.1)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
    live = st.checkbox("Live chart while running", value=True)

left, right = st.columns([1, 1])

with left:
    if st.button("Run GA", type="primary"):
        result = run_ga(
            problem=problem,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=float(crossover_rate),
            mutation_rate=float(mutation_rate),
            tournament_k=int(tournament_k),
            elitism=int(elitism),
            real_sigma=float(real_sigma),
            seed=int(seed),
            stream_live=bool(live),
        )

        st.subheader("Fitness Over Generations")
        st.line_chart(result["history"])

        st.subheader("Best Solution")
        st.write(f"Best fitness: {result['best_fitness']:.6f}")

        if problem.chromosome_type == "bit":
            bitstring = ''.join(map(str, result["best"].astype(int).tolist()))
            st.code(bitstring, language="text")
            st.write(f"Number of ones: {int(np.sum(result['best']))} / {problem.dim}")
        else:
            st.write("x* =", result["best"])

with right:
    st.subheader("Population Snapshot (final)")
    st.caption("Shows first 20 individuals with fitness")
