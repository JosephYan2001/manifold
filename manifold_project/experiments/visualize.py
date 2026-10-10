"""Replay a source Actor at a declared team size, without training or a Critic.

Retains the legacy navigation plot (numbered robots, trails and task metrics),
adapted to the entity observation/checkpoint protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manifold_project.experiments.common.runtime import configure_runtime


def save_navigation_gif(snapshots, output, fps, frame_stride, title):
    """Draw recorded native states with the legacy fixed-axis trajectory view."""
    import numpy as np
    from PIL import Image
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle
    from matplotlib import colormaps

    fig = Figure(figsize=(7, 7), dpi=100, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    paths = np.stack([s['positions'] for s in snapshots])
    goals = snapshots[0]['landmarks']
    points = np.concatenate([paths.reshape(-1, 2), goals])
    lo, hi = points.min(0) - .3, points.max(0) + .3
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal', xlabel='x', ylabel='y')
    colors = colormaps['tab10']
    circles, labels, trails = [], [], []
    for i, (pos, size) in enumerate(zip(paths[0], snapshots[0]['sizes'])):
        color = colors(i % 10)
        trails.append(ax.plot([], [], color=color, alpha=.45, lw=1)[0])
        circle = Circle(pos, size, color=color, alpha=.8)
        ax.add_patch(circle)
        circles.append(circle)
        labels.append(ax.text(*pos, str(i), ha='center', va='center', fontsize=9))
    ax.scatter(*goals.T, marker='x', s=90, color='black', label='Landmarks')
    ax.legend(loc='lower left')
    heading = ax.set_title('')
    indices = sorted(set(range(0, len(snapshots), frame_stride)) | {len(snapshots)-1})
    images = []
    try:
        for k in indices:
            state = snapshots[k]
            for i, (circle, label, trail) in enumerate(zip(circles, labels, trails)):
                circle.center = paths[k, i]
                label.set_position(paths[k, i])
                trail.set_data(paths[:k+1, i, 0], paths[:k+1, i, 1])
            heading.set_text(f"{title}\nstep {state['step']}/{snapshots[-1]['step']} | "
                             f"coverage {state['coverage']:.0%} | collision pairs {state['collision_pairs']:.0f}"
                             f"\n{state['end_reason'] or 'running'}")
            canvas.draw()
            # Store indexed images instead of retaining 1000 full RGBA canvases.
            with Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()) as rgba:
                with rgba.convert('RGB') as rgb:
                    images.append(rgb.quantize(colors=128))
        images[0].save(output, save_all=True, append_images=images[1:], loop=0,
                       duration=max(1, round(1000/fps)), optimize=False)
        return len(images)
    finally:
        for image in images:
            image.close()
        fig.clear()


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
    navigation = payload["environment"] == "navigation"
    env = make_env(payload["environment"], payload["config"], n_agents=n_agents, render_mode=None)
    before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    frames = []
    snapshots = []
    rng = np.random.default_rng(int(seed) + 170000003)
    try:
        obs, info = env.reset(seed=seed)
        def frame(step):
            if navigation:
                snapshots.append(dict(positions=np.array([a.state.p_pos.copy() for a in env.world.agents]),
                    landmarks=np.array([l.state.p_pos.copy() for l in env.world.landmarks]),
                    sizes=np.array([a.size for a in env.world.agents]), step=step,
                    coverage=info['coverage'], collision_pairs=info['collision_pairs'],
                    end_reason=info.get('end_reason')))
                return
            array = warehouse_frame(env)
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
                if step == env.horizon and not terminated and getattr(env, 'success_only', False):
                    info['end_reason'] = 'evaluation_cutoff'
                if navigation or step % frame_stride == 0 or terminated or truncated or step == env.horizon:
                    frame(step)
                if terminated or truncated:
                    break
        if any(not torch.equal(value, actor.state_dict()[key]) for key, value in before.items()):
            raise AssertionError("Replay changed frozen Actor parameters")
        output.parent.mkdir(parents=True, exist_ok=True)
        if navigation:
            count = save_navigation_gif(snapshots, output, fps, frame_stride,
                f"{payload['method']} | N={env.n_agents} | seed={seed} | "
                f"{'argmax' if deterministic else 'sampled'} policy")
        else:
            frames[0].save(output, save_all=True, append_images=frames[1:], loop=0,
                           duration=max(1, round(1000 / fps)), optimize=False)
            count = len(frames)
        return {"output": str(output), "frames": count, "scene_seed": seed,
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
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    result = render_checkpoint(args.checkpoint, args.output, args.agents, args.seed,
                               args.fps, args.deterministic, args.device, args.frame_stride)
    print(f"GIF: {result['output']} ({result['frames']} frames, frozen Actor verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
