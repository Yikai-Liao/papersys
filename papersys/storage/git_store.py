from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import polars as pl
from loguru import logger

from ..config import GitStoreConfig
from ..const import DATA_DIR
from ..fields import ID
from .summary_store import SummaryStore


class GitStore:
    """Manage an external Git repository used for JSONL artefacts."""

    TOKEN_ENV = "PAPERSYS_DATA_TOKEN"

    def __init__(self, config: GitStoreConfig) -> None:
        self.config = config
        self.repo_path = self._resolve_repo_path(config.repo_url)
        self.summary_dir = self.repo_path / config.summary_dir
        self.preference_path = self.repo_path / config.preference_file
        self.summary_store = SummaryStore(self.summary_dir)
        self.recommendation_dir = self.repo_path / "recommendations"

    # ------------------------------------------------------------------ #
    # Git operations
    # ------------------------------------------------------------------ #

    def ensure_local_copy(self) -> None:
        """Clone the repo if needed, otherwise pull the latest changes."""
        if (self.repo_path / ".git").exists():
            logger.debug("Updating existing data repo at {}", self.repo_path)
            self._run_git(["fetch", "--all"])
            self._run_git(["checkout", self.config.branch])
            self._run_git(["pull", "origin", self.config.branch])
        else:
            logger.info("Cloning data repo {} into {}", self.config.repo_url, self.repo_path)
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            clone_url = self._auth_repo_url()
            self._run_git(
                ["clone", "--branch", self.config.branch, clone_url, str(self.repo_path)],
                cwd=self.repo_path.parent,
            )

    def commit_and_push(self, message: str, paths: Sequence[Path] | None = None) -> None:
        """Commit staged changes and push to remote."""
        if paths:
            for path in paths:
                rel = path.relative_to(self.repo_path)
                self._run_git(["add", str(rel)])
        else:
            self._run_git(["add", "--all"])

        status = self._run_git(["status", "--porcelain"], capture_output=True)
        if not status.strip():
            logger.debug("No changes to commit for data repo.")
            return

        commit_msg = f"{message} ({datetime.utcnow().isoformat(timespec='seconds')}Z)"
        self._run_git(["commit", "-m", commit_msg])
        self._run_git(["push", "origin", self.config.branch])
        logger.info("Pushed data repo changes with message: {}", commit_msg)

    # ------------------------------------------------------------------ #
    # Preference helpers
    # ------------------------------------------------------------------ #

    def load_preferences(self) -> pl.DataFrame:
        """Load preference JSONL as a Polars DataFrame."""
        if not self.preference_path.exists():
            logger.warning("Preference file {} not found, returning empty DataFrame.", self.preference_path)
            return pl.DataFrame(
                {
                    ID: pl.Series(name=ID, values=[], dtype=pl.String),
                    "preference": pl.Series(name="preference", values=[], dtype=pl.String),
                }
            )
        df = pl.read_ndjson(
            self.preference_path,
            schema={ID: pl.String, "preference": pl.String},
        )
        return df

    def save_preferences(self, records: Iterable[Mapping[str, str]]) -> None:
        """Write preference records to JSONL."""
        self.preference_path.parent.mkdir(parents=True, exist_ok=True)
        with self.preference_path.open("w", encoding="utf-8") as wf:
            for record in records:
                wf.write(json.dumps(record, ensure_ascii=False))
                wf.write("\n")

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    def _resolve_repo_path(self, repo_url: str) -> Path:
        safe_name = sha1(repo_url.encode("utf-8")).hexdigest()[:10]
        base = DATA_DIR / "git_store"
        return base / safe_name

    def _auth_repo_url(self) -> str:
        token = os.getenv(self.TOKEN_ENV)
        if (
            token
            and self.config.repo_url.startswith("https://")
            and "@" not in self.config.repo_url.split("://", 1)[1]
        ):
            prefix, rest = self.config.repo_url.split("://", 1)
            return f"{prefix}://{token}@{rest}"
        return self.config.repo_url

    def _run_git(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        capture_output: bool = False,
    ) -> str:
        cmd = ["git", *args]
        run_cwd = cwd or self.repo_path
        logger.debug("Running git command: {} (cwd={})", " ".join(cmd), run_cwd)
        result = subprocess.run(
            cmd,
            cwd=run_cwd,
            capture_output=capture_output,
            text=True,
            check=True,
        )
        if capture_output:
            return (result.stdout or "").strip()
        return ""


__all__ = ["GitStore"]
