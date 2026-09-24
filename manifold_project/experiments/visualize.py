"""Replay a source Actor at a declared team size, without training or a Critic.

Retains the legacy visualize_navigation.py flow (frozen checkpoint, seeded rollout,
native task rendering, GIF), adapted to the entity observation/checkpoint protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manifold_project.experiments.common.runtime import configure_runtime


def warehouse_frame(adapter):
    """Offscreen view of native RWARE state; no window and no new task dynamics."""
    import numpy as np
    from PIL import Image, ImageDraw
    from manifold_project.experiments.applications.envs.warehouse import TINY_LAYOUT
    rows = TINY_LAYOUT.splitlines()
    cell, top = 36, 24
    image = Image.new("RGB", (len(rows[0]) * cell, len(rows) * cell + top), "white")
    draw = ImageDraw.Draw(image)
    for y, line in enumerate(rows):
        for x, value in enumerate(line):
            box = (x * cell, y * cell + top, (x + 1) * cell, (y + 1) * cell + top)
            draw.rectangle(box, fill="#b8dfc0" if value == "g" else "#f1f3f5", outline="#d2d6da")
    requests = {s.id for s in adapter.env.request_queue}
    for shelf in adapter.env.shelfs:
        x, y = int(shelf.x), int(shelf.y)
        draw.rectangle((x * cell + 7, y * cell + top + 7, (x + 1) * cell - 7, (y + 1) * cell + top - 7),
                       fill="#edb447" if shelf.id in requests else "#8b97a1", outline="#4a5158")
    for agent in adapter.env.agents:
        x, y = agent.x * cell + cell / 2, agent.y * cell + top + cell / 2
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#247ac2", outline="#123754", width=2)
        name = agent.dir.name
        dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[name]
        draw.line((x, y, x + dx * 13, y + dy * 13), fill="#112738", width=3)
    array = np.asarray(image).copy()
    image.close()
    return array


def render_checkpoint(checkpoint, output, n_agents=None, seed=1000000, fps=10,
                      deterministic=False, device="cpu", frame_stride=1):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from manifold_project.experiments.applications.evaluation import load_actor, tensor_obs
    from manifold_project.experiments.applications.envs import make_env

    if fps <= 0 or frame_stride <= 0:
        raise ValueError("fps and frame_stride must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"GIF already exists: {output}; choose another --output")
    actor, payload = load_actor(checkpoint, device)
    if payload["environment"] not in ("navigation", "warehouse"):
        raise ValueError("Task rendering is supported for navigation and warehouse checkpoints")
    env = make_env(payload["environment"], payload["config"], n_agents=n_agents,
                   render_mode=None if payload["environment"] == "warehouse" else "rgb_array")
    before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    frames = []
    rng = np.random.default_rng(int(seed) + 170000003)
    try:
        obs, info = env.reset(seed=seed)
        def frame(step):
            array = warehouse_frame(env) if payload["environment"] == "warehouse" else env.render()
            if array is None:
                raise RuntimeError("Native renderer returned no RGB frame")
            image = Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 22), fill="white")
            draw.text((5, 4), f"{payload['method']}  n={env.n_agents}  seed={seed}  t={step}  {info.get('end_reason') or 'running'}", fill="black")
            frames.append(image)
        frame(0)
        if not env.initial_done:
            for step in range(1, env.horizon + 1):
                with torch.no_grad():
                    probabilities = actor(tensor_obs(obs, device)).cpu().numpy().astype(np.float64)
                actions = probabilities.argmax(-1) if deterministic else np.array(
                    [rng.choice(len(row), p=row / row.sum()) for row in probabilities])
                obs, _, terminated, truncated, info = env.step(actions)
                if step % frame_stride == 0 or terminated or truncated:
                    frame(step)
                if terminated or truncated:
                    break
        if any(not torch.equal(value, actor.state_dict()[key]) for key, value in before.items()):
            raise AssertionError("Replay changed frozen Actor parameters")
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output, save_all=True, append_images=frames[1:], loop=0,
                       duration=max(1, round(1000 / fps)), optimize=False)
        return {"output": str(output), "frames": len(frames), "scene_seed": seed,
                "n_agents": env.n_agents, "end_reason": info.get("end_reason"),
                "frozen_actor_verified": True}
    finally:
        env.close()
        for image in frames:
            image.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen source Actor task replay; no target learning")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int)
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-duplicate-openmp", action="store_true")
    args = parser.parse_args(argv)
    configure_runtime(args.allow_duplicate_openmp)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    result = render_checkpoint(args.checkpoint, args.output, args.agents, args.seed,
                               args.fps, args.deterministic, args.device, args.frame_stride)
    print(f"GIF: {result['output']} ({result['frames']} frames, frozen Actor verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
