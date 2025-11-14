import numpy as np
import pandas as pd
import streamlit as st

# ================================================================
# Custom Problem: Bitstring length 80, max fitness at 50 ones
# ================================================================

class GAProblem:
    name = "Custom Bit Pattern"
    chromosome_type = "bit"
    dim = 80   # requirement #3
    bounds = None

    @staticmethod
    def fitness_fn(x: np.ndarray) -> float:
        ones = np.sum(x)
        return 80 - abs(ones - 50)   # requirements #2 and #4


# ================================================================
# GA Operators
# ================================================================

def init_population(pop_size: int, dim: int, rng):
    return rng.integers(0, 2, size=(pop_size, dim), dtype=np.int8)


def tournament_selection(fitness, k, rng):
    idxs = rng.integers(0, len(fitness), size=k)
    return idxs[np.argmax(fitness[idxs])]


def one_point_crossover(a, b, rng):
    point = rng.integers(1, len(a))
    return (
        np.concatenate([a[:point], b[point:]]),
        np.concatenate([b[:point], a[point:]])
    )


def mutation(x, mut_rate, rng):
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate(pop):
    return np.array([GAProblem.fitness_fn(ind) for ind in pop])


# ================================================================
# GA Runner
# ================================================================

def run_ga(
    pop_size=300,       # requirement #1
    generations=50,     # requirement #5
    crossover_rate=0.9,
    mutation_rate=0.01,
    tournament_k=3,
    elitism=2,
    seed=42,
):

    rng = np.random.default_rng(seed)

    pop = init_population(pop_size, GAProblem.dim, rng)
    fit = evaluate(pop)

    history = {"Best": [], "Average": [], "Worst": []}

    for gen in range(generations):

        # Record fitness stats
        best = np.max(fit)
        avg = np.mean(fit)
        worst = np.min(fit)

        history["Best"].append(best)
        history["Average"].append(avg)
        history["Worst"].append(worst)

        # Elitism
        elite_idx = np.argpartition(fit, -elitism)[-elitism:]
        elites = pop[elite_idx].copy()

        # Create new population
        new_pop = []
        while len(new_pop) < pop_size - elitism:

            p1 = pop[tournament_selection(fit, tournament_k, rng)]
            p2 = pop[tournament_selection(fit, tournament_k, rng)]

            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutation(c1, mutation_rate, rng)
            c2 = mutation(c2, mutation_rate, rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size - elitism:
                new_pop.append(c2)

        pop = np.vstack([new_pop, elites])
        fit = evaluate(pop)

    best_idx = np.argmax(fit)

    return {
        "best": pop[best_idx],
        "best_fitness": fit[best_idx],
        "history": pd.DataFrame(history),
        "final_population": pop,
        "final_fitness": fit,
    }


# ================================================================
# Streamlit Interface
# ================================================================

st.set_page_config(page_title="Genetic Algorithm", layout="wide")
st.title("Custom Genetic Algorithm — 80-bit Pattern (Max at 50 Ones)")

st.write("""
### Requirements Implemented:
- Population = **300**
- Chromosome length = **80 bits**
- Max fitness = **80** when ones = **50**
- Generations = **50**
""")

if st.button("Run GA", type="primary"):

    result = run_ga()

    # Store final results
    st.session_state["pop"] = result["final_population"]
    st.session_state["fit"] = result["final_fitness"]

    st.subheader("Fitness Progress")
    st.line_chart(result["history"])

    st.subheader("Best Individual")
    st.write(f"Fitness: **{result['best_fitness']}**")
    st.write(f"Number of ones: **{np.sum(result['best'])} / 80**")

    bitstring = ''.join(map(str, result["best"].astype(int)))
    st.code(bitstring)


st.subheader("Final Population Table")
if st.button("Show Table"):
    if "pop" not in st.session_state:
        st.info("Run the GA first.")
    else:
        pop = st.session_state["pop"]
        fit = st.session_state["fit"]

        df = pd.DataFrame(pop[:20])
        df["fitness"] = fit[:20]

        st.dataframe(df, use_container_width=True)
