import os
import sys
import nmmo
import importlib
from typing import Tuple, Type

from neurips2022nmmo.evaluation.team import Team

_checked = []


def load_module(submission_path: str):
    sys.path.insert(0, os.path.dirname(submission_path))
    spec = importlib.util.spec_from_file_location("submission",
                                                  submission_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_submission(submission_path: str) -> Tuple[Type, dict]:
    submission_path = os.path.join(submission_path, "submission.py")
    module = load_module(submission_path)
    return (
        module.Submission.team_klass,
        getattr(module.Submission, "init_params", {}),
    )


def get_team_from_submission(submission_path: str, team_id: str,
                             env_config: nmmo.config.Config) -> Team:
    team_klass, init_params = parse_submission(submission_path)
    if "team_id" in init_params:
        raise Exception("team_id should not be set in init_params")
    if "env_config" in init_params:
        raise Exception("env_config should not be set in init_params")
    team = team_klass(team_id, env_config, **init_params)
    return team


def check(submission_path: str) -> None:
    if submission_path in _checked:
        return

    # ray needs this
    os.environ["PYTHONPATH"] = submission_path + ":" + os.getenv(
        "PYTHONPATH", "")

    submission_path = os.path.join(submission_path, "submission.py")
    assert os.path.exists(
        submission_path), f"{submission_path} does not exist!"

    module = load_module(submission_path)

    assert hasattr(
        module,
        "Submission"), f"Submission does not exist in {submission_path}"
    assert hasattr(module.Submission,
                   "team_klass"), f"Please set team_klass in Submission."

    _checked.append(submission_path)
