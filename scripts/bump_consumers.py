"""Prepare croissant pin bumps in downstream projects after a dev tag.

croissant-sim is not on PyPI while it pins an s2fft fork, so consumers
pin a git tag instead. After pushing a dev tag, run::

    uv run python scripts/bump_consumers.py v5.3.0.dev2 --dry-run
    uv run python scripts/bump_consumers.py v5.3.0.dev2

For each consumer this checks out the consumer's default branch in a
fresh git worktree, moves the croissant-sim pin (and any s2fft fork
pin, which must match croissant's) to the tag, relocks, runs the
consumer's croissant tests against the resolved pin, and commits to
``deps/croissant-<tag>`` only if they pass. It never pushes and never
touches the consumer's own checkout: publishing a bump is a decision
made per consumer.

Consumers are listed in ``~/.config/croissant/consumers.toml``, kept
out of this repo because the paths are machine specific::

    [luseepy]
    path = "~/Documents/research/lusee/luseepy"
    test = ["uv", "run", "pytest", "tests/test_crosimulator.py"]
    env = { JAX_ENABLE_X64 = "True" }  # optional
    base = "origin/main"  # optional; defaults to origin's HEAD

A consumer must already pin croissant-sim to a git ref, either as a
``croissant-sim @ git+<url>/croissant.git@<ref>`` requirement or as a
``[tool.uv.sources]`` entry with a ``tag`` or ``rev``.
"""

import argparse
import difflib
import os
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CROISSANT = Path(__file__).resolve().parents[1]
CONFIG = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "croissant"
    / "consumers.toml"
)
WORKDIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "croissant-bump"
)
# Inherited variables that would point a consumer's tests at something
# other than the pin under test.
SCRUB_ENV = ("VIRTUAL_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT")

CROISSANT_URL_PIN = re.compile(
    r"(croissant-sim\s*@\s*git\+[^\"'\s;#]*croissant(?:\.git)?@)"
    r"([^\"'\s;#]+)"
)
CROISSANT_SOURCE_PIN = re.compile(
    r"^(croissant-sim\s*=\s*\{[^}\n]*\b(?:tag|rev)\s*=\s*\")([^\"]+)\"",
    re.MULTILINE,
)
S2FFT_URL_PIN = re.compile(r"(s2fft\s*@\s*)(git\+[^\"'\s;#]+)")
S2FFT_SOURCE_PIN = re.compile(
    r"^s2fft\s*=\s*\{[^}\n]*\bgit\s*=[^}\n]*\}", re.MULTILINE
)
BREAKING_SUBJECT = re.compile(r"^\w+(\([^)]*\))?!:")
BREAKING_FOOTER = re.compile(r"^BREAKING[ -]CHANGE:", re.MULTILINE)


@dataclass
class Rewrite:
    text: str
    old_ref: str
    s2fft_changed: bool
    notes: list[str]


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""
    breaking: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


def git(repo, *args):
    """Run git in ``repo`` and return its stdout, raising on failure."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    if proc.returncode:
        raise RuntimeError(f"git {shlex.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def git_ok(repo, *args):
    """Return whether git in ``repo`` succeeds."""
    return (
        subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True
        ).returncode
        == 0
    )


def s2fft_pin(tag):
    """Return croissant's ``git+<url>@<rev>`` s2fft pin at ``tag``."""
    project = tomllib.loads(git(CROISSANT, "show", f"{tag}:pyproject.toml"))
    for dep in project["project"]["dependencies"]:
        match = re.fullmatch(r"s2fft\s*@\s*(git\+\S+)", dep.strip())
        if match:
            return match.group(1)
    return None


def rewrite(text, tag, s2fft):
    """Move the croissant-sim pin in a pyproject ``text`` to ``tag``.

    Parameters
    ----------
    text : str
        The consumer's pyproject.toml.
    tag : str
        The croissant tag to pin.
    s2fft : str or None
        croissant's own s2fft pin at ``tag``, as ``git+<url>@<rev>``;
        None once croissant no longer pins a fork.

    Returns
    -------
    Rewrite

    Raises
    ------
    ValueError
        If croissant-sim is not pinned to a git ref, or is pinned to
        different refs in different places.
    """
    hits = [
        match
        for pattern in (CROISSANT_URL_PIN, CROISSANT_SOURCE_PIN)
        for match in pattern.finditer(text)
    ]
    refs = {match.group(2) for match in hits}
    if len(refs) != 1:
        found = ", ".join(sorted(refs)) or "none"
        raise ValueError(
            f"need one git ref pinning croissant-sim, found: {found}"
        )
    new = text
    for match in sorted(hits, key=lambda m: m.start(2), reverse=True):
        new = new[: match.start(2)] + tag + new[match.end(2) :]

    notes = []
    pinned = S2FFT_URL_PIN.search(new) or S2FFT_SOURCE_PIN.search(new)
    before = new
    if s2fft is None:
        if pinned:
            notes.append(
                "croissant no longer pins an s2fft fork; drop this "
                "consumer's s2fft pin by hand"
            )
    else:
        url, _, rev = s2fft.removeprefix("git+").rpartition("@")
        new = S2FFT_URL_PIN.sub(lambda m: m.group(1) + s2fft, new)
        new = S2FFT_SOURCE_PIN.sub(
            lambda m: f's2fft = {{ git = "{url}", rev = "{rev}" }}', new
        )
    return Rewrite(new, refs.pop(), new != before, notes)


def breaking_since(old, tag):
    """List croissant commits in ``old..tag`` marked as breaking.

    Returns
    -------
    breaking : list of str
        ``"<sha> <subject>"`` per breaking commit.
    note : str or None
        Why the range could not be checked, if it could not.
    """
    if not git_ok(CROISSANT, "rev-parse", "--verify", f"{old}^{{commit}}"):
        return [], f"{old} is unknown to croissant; breaking not checked"
    if not git_ok(CROISSANT, "merge-base", "--is-ancestor", old, tag):
        return [], f"{tag} does not descend from {old}"
    log = git(CROISSANT, "log", "--format=%h %s%x00%b%x1e", f"{old}..{tag}")
    breaking = []
    for entry in filter(str.strip, log.split("\x1e")):
        head, _, body = entry.strip().partition("\x00")
        subject = head.partition(" ")[2]
        if BREAKING_SUBJECT.match(subject) or BREAKING_FOOTER.search(body):
            breaking.append(head)
    return breaking, None


def run_logged(cmd, cwd, env, log):
    """Run ``cmd``, appending its output to ``log``; return success."""
    with log.open("a") as fh:
        fh.write(f"\n$ {shlex.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(
            cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT
        )
    return proc.returncode == 0


def commit_message(tag, change, breaking, test):
    lines = [
        f"deps: bump croissant-sim to {tag}",
        "",
        f"Moves the pin from {change.old_ref} to {tag}.",
    ]
    if change.s2fft_changed:
        lines.append("Follows croissant's s2fft fork pin at that tag.")
    if breaking:
        lines += ["", "croissant commits marked breaking since the old pin:"]
        lines += [f"- {commit}" for commit in breaking]
    lines += [
        "",
        "Prepared by croissant's scripts/bump_consumers.py; this passed",
        "against the new pin:",
        "",
        f"    {shlex.join(test)}",
    ]
    return "\n".join(lines) + "\n"


def bump(name, cfg, tag, s2fft, args):
    """Prepare (or, with ``--dry-run``, preview) one consumer's bump."""
    say = lambda msg: print(f"[{name}] {msg}", flush=True)  # noqa: E731
    repo = Path(cfg["path"]).expanduser()
    if args.fetch:
        say("fetching origin")
        git(repo, "fetch", "--quiet", "origin")
    base = cfg.get("base")
    if base is None:
        if not git_ok(repo, "symbolic-ref", "refs/remotes/origin/HEAD"):
            raise RuntimeError(
                "origin/HEAD is unset; set `base` in the config or run "
                "`git remote set-head origin --auto`"
            )
        base = git(
            repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"
        ).strip()

    original = git(repo, "show", f"{base}:pyproject.toml")
    try:
        change = rewrite(original, tag, s2fft)
    except ValueError as err:
        return Result(name, "failed", f"{base}: {err}")
    result = Result(name, "planned", notes=change.notes)
    if change.text == original:
        result.status = "up to date"
        result.detail = f"{base} already pins {tag}"
        return result
    result.detail = f"{change.old_ref} -> {tag} on {base}"
    result.breaking, note = breaking_since(change.old_ref, tag)
    if note:
        result.notes.append(note)

    if args.dry_run:
        sys.stdout.writelines(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                change.text.splitlines(keepends=True),
                f"{name}/pyproject.toml ({base})",
                f"{name}/pyproject.toml ({tag})",
            )
        )
        return result

    branch = f"deps/croissant-{tag}"
    tree = args.workdir / name / tag
    log = args.workdir / name / f"{tag}.log"
    if git_ok(repo, "rev-parse", "--verify", f"refs/heads/{branch}"):
        result.status = "skipped"
        result.detail = f"branch {branch} already exists"
        return result
    if tree.exists():
        result.status = "skipped"
        result.detail = f"{tree} already exists"
        return result
    tree.parent.mkdir(parents=True, exist_ok=True)
    log.unlink(missing_ok=True)
    say(f"worktree {tree} on new branch {branch}")
    git(repo, "worktree", "add", "--quiet", "-b", branch, str(tree), base)
    (tree / "pyproject.toml").write_text(change.text)

    env = {k: v for k, v in os.environ.items() if k not in SCRUB_ENV}
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    cleanup = (
        f"git -C {repo} worktree remove --force {tree} && "
        f"git -C {repo} branch -D {branch}"
    )
    lock = ["uv", "lock", "--upgrade-package", "croissant-sim"]
    if change.s2fft_changed:
        lock += ["--upgrade-package", "s2fft"]
    say("resolving the new pin")
    if not run_logged(lock, tree, env, log):
        result.status = "failed"
        result.detail = f"uv lock failed; see {log}"
        result.next_steps.append(f"discard: {cleanup}")
        return result
    say(f"testing (log: {log})")
    if not run_logged(cfg["test"], tree, env, log):
        result.status = "failed"
        result.detail = f"tests failed, left uncommitted; see {log}"
        result.next_steps.append(f"discard: {cleanup}")
        return result

    files = ["pyproject.toml"]
    if git_ok(tree, "ls-files", "--error-unmatch", "uv.lock"):
        files.append("uv.lock")
    git(tree, "add", *files)
    message = commit_message(tag, change, result.breaking, cfg["test"])
    git(tree, "commit", "--quiet", "-m", message)
    result.status = "committed"
    result.next_steps += [
        f"push: git -C {repo} push -u origin {branch}",
        f"then: git -C {repo} worktree remove {tree}",
    ]
    return result


def load_config(path, only):
    """Read and validate the consumer table, keeping ``only`` if set."""
    if not path.exists():
        sys.exit(f"no consumer config at {path}; see this script's doc")
    consumers = tomllib.loads(path.read_text())
    for name, cfg in consumers.items():
        if "path" not in cfg or "test" not in cfg:
            sys.exit(f"{path}: [{name}] needs `path` and `test`")
        if isinstance(cfg["test"], str):
            cfg["test"] = shlex.split(cfg["test"])
    unknown = set(only or ()) - set(consumers)
    if unknown:
        sys.exit(f"not in {path}: {', '.join(sorted(unknown))}")
    return {n: c for n, c in consumers.items() if not only or n in only}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tag", help="croissant tag to pin, e.g. v5.3.0.dev2")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show each consumer's pin diff; create nothing",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help="consumers to bump"
    )
    parser.add_argument(
        "--no-fetch",
        dest="fetch",
        action="store_false",
        help="use the consumers' remote-tracking refs as they are",
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=WORKDIR,
        help="where worktrees and logs go (default: %(default)s)",
    )
    args = parser.parse_args()
    tag = args.tag

    if not git_ok(CROISSANT, "rev-parse", "--verify", f"refs/tags/{tag}"):
        sys.exit(f"{tag} is not a tag in {CROISSANT}")
    local = git(CROISSANT, "rev-parse", f"refs/tags/{tag}").strip()
    remote = git(CROISSANT, "ls-remote", "origin", f"refs/tags/{tag}")
    if remote.split()[:1] != [local]:
        problem = "is not on origin" if not remote else "differs on origin"
        if not args.dry_run:
            sys.exit(f"{tag} {problem}; consumers resolve it from GitHub")
        print(f"warning: {tag} {problem}", file=sys.stderr)
    consumers = load_config(args.config, args.only)
    s2fft = s2fft_pin(tag)

    results = []
    for name, cfg in consumers.items():
        try:
            results.append(bump(name, cfg, tag, s2fft, args))
        except RuntimeError as err:
            results.append(Result(name, "failed", str(err)))

    print(f"\n{tag}{' (dry run)' if args.dry_run else ''}:")
    for r in results:
        print(f"  {r.name:<16} {r.status:<11} {r.detail}")
        for commit in r.breaking:
            print(f"      breaking: {commit}")
        for line in r.notes + r.next_steps:
            print(f"      {line}")
    if not args.dry_run:
        print("\nNothing was pushed.")
    return int(any(r.status == "failed" for r in results))


if __name__ == "__main__":
    sys.exit(main())
