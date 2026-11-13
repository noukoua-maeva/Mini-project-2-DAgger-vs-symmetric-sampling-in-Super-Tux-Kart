"""Solo SuperTuxKart pilot and comparison. Requires the course repository's src on PYTHONPATH."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agents.team3.agent3 import Agent3
from pystk2_gymnasium.envs import AgentSpec, STKRaceEnv

DT = 0.1
HORIZON = 1000
TRACK = "lighthouse"
BUDGET = 24_000
CHECKPOINT = 4_000
BATCH = 128
DEMO_SEEDS = range(100, 110)
EVAL_SEEDS = range(500, 503)
ACTIONS = ("steer", "acceleration", "brake", "drift", "nitro", "rescue")


def encode(obs):
    paths = np.zeros((8, 3), dtype=np.float32)
    for i, (start, width) in enumerate(zip(obs["paths_start"], obs["paths_width"])):
        paths[i] = [start[0] / 20, start[2] / 40, float(np.asarray(width).item()) / 10]
    features = [*paths.ravel(), *(np.asarray(obs["velocity"]) / 30)]
    for key, scale in (("center_path_distance", 5), ("max_steer_angle", 1),
                       ("energy", 5), ("skeed_factor", 3), ("shield_time", 5),
                       ("distance_down_track", 1000)):
        features.append(float(np.asarray(obs[key]).item()) / scale)
    items = np.zeros((4, 4), dtype=np.float32)
    for i, (pos, kind) in enumerate(zip(obs["items_position"][:4], obs["items_type"][:4])):
        items[i] = [pos[0] / 10, pos[2] / 40, kind == 1, kind == 4]
    return np.array([*features, *items.ravel()], dtype=np.float32)


def expert_vector(expert, obs):
    action = expert.choose_action(obs)
    return np.array([action["steer"], 2 * action["acceleration"] - 1,
                     *[2 * bool(action[key]) - 1 for key in ACTIONS[2:]]], np.float32)


def decode(vector):
    return dict(steer=float(vector[0]), acceleration=float((vector[1] + 1) / 2),
                **{k: bool(v > 0) for k, v in zip(ACTIONS[2:], vector[2:])}, fire=False)


def transition(env, action):
    obs, reward, done, _, info = env.step(decode(action))
    return obs, float(reward), bool(done), info


def make_env(track):
    return STKRaceEnv(track=track, num_kart=1, laps=1, max_paths=8, step_size=DT,
                      agent=AgentSpec(rank_start=0), use_subprocess=False)


def episode(env, seed, actor=None, dataset=None):
    obs, _ = env.reset(seed=seed)
    expert = Agent3(env) if actor is None else None
    for step in range(1, HORIZON + 1):
        x = encode(obs)
        action = expert_vector(expert, obs) if expert is not None else actor(x)
        obs, reward, done, info = transition(env, action)
        if dataset is not None:
            dataset.add(x, action, reward, encode(obs), done)
        if done:
            break
    finish_time = float(env.world.karts[0].finish_time) if done else None
    return step, done, float(info["distance"]), finish_time


class Replay:
    def __init__(self, size, obs_dim):
        self.x = np.empty((size, obs_dim), np.float32)
        self.a = np.empty((size, len(ACTIONS)), np.float32)
        self.r = np.empty((size, 1), np.float32)
        self.y = np.empty_like(self.x)
        self.d = np.empty_like(self.r)
        self.count = 0

    def add(self, x, a, r, y, d):
        i = self.count % len(self.x)
        self.x[i], self.a[i], self.r[i], self.y[i], self.d[i] = x, a, r, y, d
        self.count += 1

    def sample(self, rng, count):
        i = rng.integers(min(self.count, len(self.x)), size=count)
        return tuple(torch.from_numpy(arr[i]) for arr in (self.x, self.a, self.r, self.y, self.d))


def network(inputs, outputs, policy=False):
    layers = [nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
              nn.Linear(128, outputs)]
    if policy:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Agent:
    def __init__(self, obs_dim):
        self.actor = network(obs_dim, len(ACTIONS), policy=True)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

    @torch.no_grad()
    def action(self, x):
        return self.actor(torch.from_numpy(x)).numpy()

    def imitate(self, batch):
        x, a, *_ = batch
        prediction = self.actor(x)
        loss = (prediction[:, :2] - a[:, :2]).square().mean()
        # Tanh outputs for switches are thresholded at zero at execution.
        loss = loss + .25 * (prediction[:, 2:] - a[:, 2:]).square().mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()


class TD3(Agent):
    def __init__(self, obs_dim):
        super().__init__(obs_dim)
        import copy
        self.target_actor = copy.deepcopy(self.actor)
        self.critics = nn.ModuleList([network(obs_dim + len(ACTIONS), 1) for _ in range(2)])
        self.targets = copy.deepcopy(self.critics)
        self.optimizer = torch.optim.Adam(self.critics.parameters(), lr=3e-4)
        self.updates = 0

    def update(self, batch):
        x, a, r, y, d = batch
        self.updates += 1
        with torch.no_grad():
            noise = (.2 * torch.randn_like(a)).clamp(-.5, .5)
            next_a = (self.target_actor(y) + noise).clamp(-1, 1)
            next_xa = torch.cat((y, next_a), dim=-1)
            target = r + .99 * (1 - d) * torch.minimum(*(q(next_xa) for q in self.targets))
        xa = torch.cat((x, a), dim=-1)
        loss = sum((q(xa) - target).square().mean() for q in self.critics)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.updates % 2 == 0:
            actor_loss = -self.critics[0](torch.cat((x, self.actor(x)), dim=-1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            with torch.no_grad():
                for target_model, model in ((self.target_actor, self.actor), (self.targets, self.critics)):
                    for target_param, param in zip(target_model.parameters(), model.parameters()):
                        target_param.lerp_(param, .005)


def train(env, demos, writer, file, method, seed, budget):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    obs_dim = demos.x.shape[1]
    replay = Replay(budget + demos.count, obs_dim)
    if method == "DAgger":
        agent = Agent(obs_dim)
        for i in range(demos.count):
            replay.add(demos.x[i], demos.a[i], 0, demos.y[i], False)
        for _ in range(1000):
            agent.imitate(replay.sample(rng, BATCH))
    else:
        agent = TD3(obs_dim)

    def evaluate(step):
        for eval_seed in EVAL_SEEDS:
            ticks, finished, distance, time = episode(env, eval_seed, agent.action)
            writer.writerow([method, seed, step, eval_seed, ticks, int(finished), distance, time])
        file.flush()
        print(method, seed, step, flush=True)

    # Evaluate at fixed checkpoints, then begin a new training episode.
    evaluate(0)
    race_index = 0
    obs, _ = env.reset(seed=10_000 + seed * 100 + race_index)
    expert = Agent3(env)
    race_steps = 0
    for step in range(1, budget + 1):
        x = encode(obs)
        action = agent.action(x)
        if method == "DAgger":
            label = expert_vector(expert, obs)
            beta = max(0.0, .5 * (1 - (step - 1) // CHECKPOINT / 3))
            if rng.random() < beta:
                action = label
        else:
            action = np.clip(action + rng.normal(0, .15, len(ACTIONS)), -1, 1).astype(np.float32)
        obs, reward, done, _ = transition(env, action)
        race_steps += 1
        replay.add(x, label if method == "DAgger" else action, reward, encode(obs), done)
        if method == "DAgger":
            agent.imitate(replay.sample(rng, BATCH))
        elif step >= BATCH:
            if method == "TD3 50/50":
                batch = tuple(torch.cat((a, b)) for a, b in zip(
                    demos.sample(rng, BATCH // 2), replay.sample(rng, BATCH // 2)))
            else:
                batch = replay.sample(rng, BATCH)
            agent.update(batch)
        checkpoint = step % CHECKPOINT == 0 or step == budget
        if checkpoint:
            evaluate(step)
        if (done or race_steps == HORIZON or checkpoint) and step < budget:
            race_index += 1
            obs, _ = env.reset(seed=10_000 + seed * 100 + race_index)
            expert = Agent3(env)
            race_steps = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=TRACK)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--method", choices=("DAgger", "TD3 50/50", "TD3 online"), default="DAgger")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    env = make_env(args.track)
    try:
        x = encode(env.reset(seed=0)[0])
        demos = Replay(len(DEMO_SEEDS) * HORIZON, len(x))
        with args.output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["method", "seed", "step", "race_seed", "ticks", "finished", "distance", "finish_time_s"])
            for seed in DEMO_SEEDS:
                ticks, done, distance, time = episode(env, seed, dataset=demos)
                writer.writerow(["Demonstration", 0, 0, seed, ticks, int(done), distance, time])
            for seed in EVAL_SEEDS:
                ticks, done, distance, time = episode(env, seed)
                writer.writerow(["Expert", 0, 0, seed, ticks, int(done), distance, time])
            file.flush()
            train(env, demos, writer, file, args.method, args.seed, args.budget)
    finally:
        env.close()


if __name__ == "__main__":
    main()
