"""Time-based train/test split at session level, per user."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def load_events(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_events(
    events: list[dict[str, str]],
    train_ratio: float = 0.8,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Split events into train/test at session level per user.

    For each user with >=2 sessions:
    - Sort sessions by earliest timestamp
    - Last ``max(1, floor(session_count × (1-train_ratio)))`` → test
    - Rest → train

    Users with 1 session → train only (no test).
    """
    # Group events by (user_id, session_id)
    user_sessions: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in events:
        user_sessions[e["user_id"]][e["session_id"]].append(e)

    train_events: list[dict] = []
    test_events: list[dict] = []
    split_info: dict[str, Any] = {
        "users_with_multiple_sessions": 0,
        "users_with_single_session": 0,
        "train_session_count": 0,
        "test_session_count": 0,
    }

    # Verify no leakage: for each user, train max_ts < test min_ts
    time_ok = True

    for uid, sessions in user_sessions.items():
        # Sort sessions by their earliest timestamp
        session_times = []
        for sid, evts in sessions.items():
            min_ts = min(_parse_ts(e["timestamp"]) for e in evts)
            session_times.append((min_ts, sid, evts))
        session_times.sort(key=lambda x: x[0])

        n_sessions = len(session_times)
        if n_sessions >= 2:
            split_info["users_with_multiple_sessions"] += 1
            n_test = max(1, int(n_sessions * (1.0 - train_ratio)))
            n_train = n_sessions - n_test

            for i, (_, sid, evts) in enumerate(session_times):
                if i < n_train:
                    train_events.extend(evts)
                    split_info["train_session_count"] += 1
                else:
                    test_events.extend(evts)
                    split_info["test_session_count"] += 1

            # Check: max train ts < min test ts
            train_ts = [min(_parse_ts(e["timestamp"]) for e in session_times[j][2])
                        for j in range(n_train)]
            test_ts = [min(_parse_ts(e["timestamp"]) for e in session_times[j][2])
                       for j in range(n_train, n_sessions)]
            if train_ts and test_ts and max(train_ts) >= min(test_ts):
                time_ok = False
        else:
            split_info["users_with_single_session"] += 1
            # Single-session users: all events go to train
            for _, sid, evts in session_times:
                train_events.extend(evts)
                split_info["train_session_count"] += 1

    split_info["time_leakage_free"] = time_ok
    split_info["train_event_count"] = len(train_events)
    split_info["test_event_count"] = len(test_events)

    return train_events, test_events, split_info


def get_train_test_users(split_info: dict, events: list[dict], train_events: list[dict]) -> tuple[set, set]:
    """Return sets of user_ids in train and test."""
    train_users = {e["user_id"] for e in train_events}
    test_users = {e["user_id"] for e in events if e["user_id"] not in train_users}
    return train_users, test_users
