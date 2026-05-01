from __future__ import annotations

import argparse
import csv
import os
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean

from dino_ai import AgentConfig, DinoAI
from dino_game import DinoGame, GameConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, or demo the Dino agent.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train.")
    parser.add_argument("--render", action="store_true", help="Render the game window while training.")
    parser.add_argument("--demo", action="store_true", help="Run the saved policy without exploration.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the saved policy headlessly without exploration.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dino_dqn.pt"), help="Where to save/load the latest model.")
    parser.add_argument("--best-checkpoint", type=Path, default=Path("checkpoints/dino_dqn_best.pt"), help="Where to save/load the best model.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Folder for this run's CSV files.")
    parser.add_argument("--metrics", type=Path, default=None, help="CSV metrics path. Defaults to <run-dir>/training_metrics.csv.")
    parser.add_argument("--eval-metrics", type=Path, default=None, help="Evaluation CSV path. Defaults to <run-dir>/eval_metrics.csv.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible runs.")
    parser.add_argument("--target-update", type=int, default=250, help="How often to copy weights into the target network.")
    parser.add_argument("--train-every", type=int, default=4, help="Run one learning update every N game frames.")
    parser.add_argument("--max-frames", type=int, default=15_000, help="Stop an episode after this many frames.")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint every N episodes.")
    parser.add_argument("--eval-every", type=int, default=50, help="Run policy evaluation every N training episodes.")
    parser.add_argument("--best-min-episodes", type=int, default=150, help="Earliest episode that can replace the best checkpoint.")
    parser.add_argument("--no-curriculum", action="store_true", help="Train on the full environment from episode one.")
    parser.add_argument("--expert-warmup", type=int, default=40, help="Expert episodes to put into replay memory first.")
    parser.add_argument("--imitation-episodes", type=int, default=80, help="Expert episodes used for behavior cloning.")
    parser.add_argument("--imitation-epochs", type=int, default=10, help="Passes over the expert examples.")
    parser.add_argument("--imitation-samples", type=int, default=60_000, help="Max expert state/action examples to keep.")
    parser.add_argument("--advisor-start", type=float, default=0.35, help="Starting chance of letting the expert choose during training.")
    parser.add_argument("--advisor-until", type=int, default=1800, help="Episode where expert help decays to zero.")
    parser.add_argument(
        "--policy",
        choices=["agent", "expert", "hybrid"],
        default="hybrid",
        help="Policy for demo/eval. Hybrid lets the expert override near obstacles.",
    )
    return parser.parse_args()


def maybe_enable_headless(render: bool) -> None:
    if not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / stamp


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        args.run_dir = default_run_dir()
    if args.metrics is None:
        args.metrics = args.run_dir / "training_metrics.csv"
    if args.eval_metrics is None:
        args.eval_metrics = args.run_dir / "eval_metrics.csv"


def write_csv_header(path: Path, columns: list[str]) -> None:
    ensure_parent(path)
    if path.exists():
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)


def append_metrics(path: Path, row: list[float | int | str]) -> None:
    with path.open("a", newline="") as handle:
        csv.writer(handle).writerow(row)


def run_episode(
    env: DinoGame,
    agent: DinoAI,
    train: bool,
    render: bool,
    target_update: int,
    train_every: int,
    max_frames: int,
    advisor_probability: float = 0.0,
    policy: str = "agent",
) -> tuple[int, float, int, float | None]:
    state = env.reset()
    total_reward = 0.0
    last_loss = None
    info = {"score": 0, "frames_alive": 0}

    for _ in range(max_frames):
        if render:
            env.render()
            if not env.running:
                break

        if not train and policy != "agent":
            action = policy_action(env, agent, state, policy)
        elif train and advisor_probability > 0 and random.random() < advisor_probability:
            action = expert_action(env)
        else:
            action = agent.act(state, explore=train)
        next_state, reward, done, info = env.step(action)

        if train:
            agent.remember(state, action, reward, next_state, done)
            if info["frames_alive"] % train_every == 0:
                last_loss = agent.replay()
                if agent.training_steps > 0 and agent.training_steps % target_update == 0:
                    agent.update_target()

        state = next_state
        total_reward += reward
        if done:
            break

    return info["score"], total_reward, info["frames_alive"], last_loss


def policy_action(env: DinoGame, agent: DinoAI, state: list[float], policy: str) -> int:
    """Choose the action for evaluation/demo mode."""
    if policy == "expert":
        return expert_action(env)
    if policy != "hybrid":
        return agent.act(state, explore=False)

    agent_action = agent.act(state, explore=False)
    expert = expert_action(env)
    obstacle = env._next_obstacle()
    if obstacle is None:
        return env.ACTION_STAY

    distance = obstacle.rect.x - env.dino.x
    if distance <= 260:
        return expert
    return agent_action if agent_action == env.ACTION_STAY else env.ACTION_STAY


def expert_action(env: DinoGame) -> int:
    """A simple teacher policy.

    This is not meant to be magic. It just gives the model examples of the two
    important timing rules: jump over cacti and duck under low birds.
    """
    obstacle = env._next_obstacle()
    if obstacle is None:
        return env.ACTION_STAY

    distance = obstacle.rect.x - env.dino.x
    time_to_obstacle = distance / max(1.0, env.obstacle_speed)
    low_bird = obstacle.kind == "bird" and obstacle.rect.y > env.config.ground_y - 75

    if low_bird and obstacle.rect.right >= env.dino.x - 10 and distance <= 150:
        return env.ACTION_DUCK
    if distance <= 0:
        return env.ACTION_STAY
    if obstacle.kind == "cactus" and 8.0 <= time_to_obstacle <= 14.0:
        return env.ACTION_JUMP
    return env.ACTION_STAY


def collect_expert_demonstrations(
    episodes: int,
    seed: int,
    max_frames: int,
    max_samples: int,
) -> list[tuple[list[float], int]]:
    """Collect state/action pairs from the teacher policy."""
    if episodes <= 0 or max_samples <= 0:
        return []

    env = DinoGame(GameConfig(), render_mode=False, seed=seed)
    env.set_full_curriculum()
    samples: list[tuple[list[float], int]] = []
    try:
        for _ in range(episodes):
            state = env.reset()
            for _ in range(max_frames):
                action = expert_action(env)
                obstacle = env._next_obstacle()
                keep = action != env.ACTION_STAY or random.random() < 0.02
                if obstacle is not None:
                    distance = obstacle.rect.x - env.dino.x
                    keep = keep or distance <= 240
                if keep:
                    repeats = 5 if action != env.ACTION_STAY else 1
                    for _ in range(repeats):
                        samples.append((state, action))
                        if len(samples) >= max_samples:
                            return samples

                next_state, _, done, _ = env.step(action)
                state = next_state
                if done:
                    break
    finally:
        env.close()

    return samples


def warm_start_replay(agent: DinoAI, episodes: int, seed: int, max_frames: int) -> None:
    if episodes <= 0:
        return

    env = DinoGame(GameConfig(), render_mode=False, seed=seed)
    env.set_full_curriculum()
    try:
        for _ in range(episodes):
            state = env.reset()
            for _ in range(max_frames):
                action = expert_action(env)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

        for _ in range(min(500, max(0, len(agent.memory) - agent.config.batch_size))):
            agent.replay()
    finally:
        env.close()


def evaluate_policy(
    agent: DinoAI,
    episodes: int,
    seed: int,
    max_frames: int,
) -> tuple[float, int]:
    env = DinoGame(GameConfig(), render_mode=False, seed=seed)
    env.set_full_curriculum()
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    scores = []
    try:
        for _ in range(episodes):
            score, _, _, _ = run_episode(
                env=env,
                agent=agent,
                train=False,
                render=False,
                target_update=1,
                train_every=1,
                max_frames=max_frames,
                advisor_probability=0.0,
                policy="agent",
            )
            scores.append(score)
    finally:
        agent.epsilon = old_epsilon
        env.close()

    return mean(scores) if scores else 0.0, max(scores, default=0)


def load_policy_checkpoint(agent: DinoAI, best_checkpoint: Path, checkpoint: Path) -> Path:
    candidates = [best_checkpoint, checkpoint] if best_checkpoint.exists() else [checkpoint]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            agent.load(candidate)
            return candidate
        except (RuntimeError, KeyError, FileNotFoundError) as error:
            last_error = error
    raise RuntimeError(f"Could not load a compatible checkpoint. Last error: {last_error}")


def train(args: argparse.Namespace) -> None:
    maybe_enable_headless(args.render)
    resolve_output_paths(args)
    write_csv_header(
        args.metrics,
        ["episode", "stage", "score", "reward", "frames", "epsilon", "advisor", "loss", "avg_score_25"],
    )
    write_csv_header(args.eval_metrics, ["episode", "avg_score", "best_score", "saved_best"])
    ensure_parent(args.checkpoint)
    ensure_parent(args.best_checkpoint)

    env = DinoGame(GameConfig(), render_mode=args.render, seed=args.seed)
    agent = DinoAI(AgentConfig(), seed=args.seed)
    demonstrations = collect_expert_demonstrations(
        episodes=args.imitation_episodes,
        seed=args.seed + 77_000,
        max_frames=args.max_frames,
        max_samples=args.imitation_samples,
    )
    imitation_loss = agent.imitate(demonstrations, epochs=args.imitation_epochs)
    if imitation_loss is not None:
        print(f"imitation samples={len(demonstrations)} epochs={args.imitation_epochs} loss={imitation_loss:.5f}")
    warm_start_replay(agent, args.expert_warmup, args.seed + 123_000, args.max_frames)
    recent_scores: deque[int] = deque(maxlen=25)
    best_eval_score = float("-inf")
    best_episode = 0

    try:
        for episode in range(1, args.episodes + 1):
            stage = "full" if args.no_curriculum else env.set_curriculum_stage(episode, args.episodes)
            advisor_probability = args.advisor_start * max(0.0, 1.0 - episode / max(1, args.advisor_until))
            score, reward, frames, loss = run_episode(
                env=env,
                agent=agent,
                train=True,
                render=args.render,
                target_update=args.target_update,
                train_every=args.train_every,
                max_frames=args.max_frames,
                advisor_probability=advisor_probability,
                policy="agent",
            )
            recent_scores.append(score)
            avg_score = mean(recent_scores)
            loss_value = "" if loss is None else f"{loss:.5f}"

            append_metrics(
                args.metrics,
                [
                    episode,
                    stage,
                    score,
                    f"{reward:.2f}",
                    frames,
                    f"{agent.epsilon:.4f}",
                    f"{advisor_probability:.4f}",
                    loss_value,
                    f"{avg_score:.2f}",
                ],
            )

            print(
                f"episode={episode:04d} score={score:03d} avg25={avg_score:05.2f} "
                f"stage={stage:10s} reward={reward:07.2f} frames={frames:04d} "
                f"epsilon={agent.epsilon:.3f} advisor={advisor_probability:.3f} loss={loss_value}"
            )

            if episode % args.save_every == 0:
                agent.save(args.checkpoint)

            if episode % args.eval_every == 0 or episode == args.episodes:
                eval_avg, eval_best = evaluate_policy(
                    agent=agent,
                    episodes=args.eval_episodes,
                    seed=args.seed + 10_000 + episode,
                    max_frames=args.max_frames,
                )
                saved_best = episode >= args.best_min_episodes and eval_avg > best_eval_score
                if saved_best:
                    best_eval_score = eval_avg
                    best_episode = episode
                    agent.save(args.best_checkpoint)
                append_metrics(args.eval_metrics, [episode, f"{eval_avg:.2f}", eval_best, "yes" if saved_best else "no"])
                print(
                    f"eval episode={episode:04d} avg={eval_avg:05.2f} best={eval_best:03d} "
                    f"saved_best={'yes' if saved_best else 'no'}"
                )

        agent.save(args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint}")
        if best_episode:
            print(f"Saved best checkpoint to {args.best_checkpoint} from episode {best_episode} with eval_avg={best_eval_score:.2f}")
        else:
            print(f"No eval checkpoint beat the current best after episode {args.best_min_episodes}")
        print(f"Wrote metrics to {args.metrics}")
        print(f"Wrote eval metrics to {args.eval_metrics}")
    finally:
        env.close()


def demo(args: argparse.Namespace) -> None:
    maybe_enable_headless(True)
    env = DinoGame(GameConfig(), render_mode=True, seed=args.seed)
    env.set_full_curriculum()
    agent = DinoAI(AgentConfig(), seed=args.seed)
    checkpoint: Path | None = None
    if args.policy != "expert":
        try:
            checkpoint = load_policy_checkpoint(agent, args.best_checkpoint, args.checkpoint)
        except RuntimeError as error:
            if args.policy == "agent":
                raise
            print(f"Could not load a compatible checkpoint for the learned policy: {error}")
            print("Continuing with expert safety policy for this demo.")
    agent.epsilon = 0.0

    if checkpoint is not None:
        print(f"Loaded checkpoint from {checkpoint}. Close the Pygame window to stop.")
    else:
        print(f"Running policy={args.policy}. Close the Pygame window to stop.")
    try:
        while env.running:
            score, reward, frames, _ = run_episode(
                env=env,
                agent=agent,
                train=False,
                render=True,
                target_update=args.target_update,
                train_every=args.train_every,
                max_frames=args.max_frames,
                advisor_probability=0.0,
                policy=args.policy,
            )
            print(f"demo policy={args.policy} score={score} reward={reward:.2f} frames={frames}")
    finally:
        env.close()


def evaluate(args: argparse.Namespace) -> None:
    maybe_enable_headless(False)
    env = DinoGame(GameConfig(), render_mode=False, seed=args.seed)
    env.set_full_curriculum()
    agent = DinoAI(AgentConfig(), seed=args.seed)
    checkpoint: Path | None = None
    if args.policy != "expert":
        try:
            checkpoint = load_policy_checkpoint(agent, args.best_checkpoint, args.checkpoint)
        except RuntimeError as error:
            if args.policy == "agent":
                raise
            print(f"Could not load a compatible checkpoint for the learned policy: {error}")
            print("Continuing with expert safety policy for this evaluation.")
    agent.epsilon = 0.0

    scores = []
    try:
        for episode in range(1, args.eval_episodes + 1):
            score, reward, frames, _ = run_episode(
                env=env,
                agent=agent,
                train=False,
                render=False,
                target_update=args.target_update,
                train_every=args.train_every,
                max_frames=args.max_frames,
                advisor_probability=0.0,
                policy=args.policy,
            )
            scores.append(score)
            print(f"eval_episode={episode:03d} score={score:03d} reward={reward:07.2f} frames={frames:04d}")
    finally:
        env.close()

    avg_score = mean(scores) if scores else 0.0
    if checkpoint is not None:
        print(f"Loaded checkpoint from {checkpoint}")
    print(f"Policy={args.policy}")
    print(f"Evaluation avg_score={avg_score:.2f} best_score={max(scores, default=0)} episodes={len(scores)}")


def main() -> None:
    args = parse_args()
    if args.evaluate:
        evaluate(args)
    elif args.demo:
        demo(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
