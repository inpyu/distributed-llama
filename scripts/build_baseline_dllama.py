#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys


BASELINE_COMMIT = "d2c185e1f1335047e6ee3fd5046a09399dd4f515"


def run(cmd: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def capture(cmd: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), text=True)


def ensure_git_repo(root: Path) -> None:
    try:
        out = capture(["git", "rev-parse", "--is-inside-work-tree"], cwd=root).strip()
    except Exception as exc:
        raise RuntimeError("Not a git repository (or git not available).") from exc
    if out != "true":
        raise RuntimeError("Not inside a git work tree.")


def worktree_exists(root: Path, worktree_dir: Path) -> bool:
    porcelain = capture(["git", "worktree", "list", "--porcelain"], cwd=root)
    needle = f"worktree {worktree_dir}\n"
    return needle in porcelain


def ensure_worktree(root: Path, worktree_dir: Path, commit: str) -> None:
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    if worktree_exists(root, worktree_dir):
        return
    run(["git", "worktree", "add", "--detach", str(worktree_dir), commit], cwd=root)


def make_baseline(worktree_dir: Path, target: str, clean: bool) -> None:
    if clean:
        run(["make", "clean"], cwd=worktree_dir)
    run(["make", target], cwd=worktree_dir)


def chmod_x(path: Path) -> None:
    if os.name != "posix":
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a baseline dllama binary from a pinned commit into a separate output name.",
    )
    parser.add_argument("--commit", default=BASELINE_COMMIT, help="Git commit to build.")
    parser.add_argument("--target", default="dllama", help="Make target to build in the baseline worktree.")
    parser.add_argument("--out", default="dllama_0", help="Output binary name in the current repo root.")
    parser.add_argument(
        "--worktree-dir",
        default=None,
        help="Worktree directory (default: .worktrees/<out>_<commit8>).",
    )
    parser.add_argument("--no-clean", action="store_true", help="Skip `make clean` in the baseline worktree.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ensure_git_repo(root)

    commit8 = str(args.commit)[:8]
    out_path = root / args.out
    worktree_dir = (
        Path(args.worktree_dir).resolve()
        if args.worktree_dir
        else (root / ".worktrees" / f"{args.out}_{commit8}")
    )

    ensure_worktree(root, worktree_dir, str(args.commit))
    make_baseline(worktree_dir, str(args.target), clean=(not args.no_clean))

    built_path = worktree_dir / args.target
    if not built_path.exists():
        raise RuntimeError(f"Expected build output not found: {built_path}")

    shutil.copy2(built_path, out_path)
    chmod_x(out_path)
    print(f"Built {out_path} from {args.commit}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
