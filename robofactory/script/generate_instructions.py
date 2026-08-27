"""Generate the per-task natural language instruction bank used by CLS-DP.

The paper generates task-level instructions with an LLM and then diversifies them by
sampling alternative phrasings (following RoboTwin 2.0), producing 100 training and 100
held-out evaluation instructions per task. We reproduce that protocol with a deterministic
combinatorial generator so the data pipeline stays offline and reproducible.

The seed phrasings below are taken from Table IV of the paper. Object colours are taken
from this repo's task configs rather than the paper, because they disagree: the paper's
stacking instructions say "blue/red/green" while the configs define cubeA blue, cubeB
green, cubeC red.

Usage:
    python script/generate_instructions.py --task_name LiftBarrier
    python script/generate_instructions.py --all
"""

import argparse
import itertools
import json
import os
import random

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(_PACKAGE_DIR, "configs", "instructions")

N_TRAIN = 100
N_EVAL = 100


def _expand(templates, slots):
    """Expand every template against the cartesian product of its slot values."""
    phrasings = []
    for template in templates:
        keys = [k for k in slots if "{" + k + "}" in template]
        if not keys:
            phrasings.append(template)
            continue
        for combo in itertools.product(*[slots[k] for k in keys]):
            phrasings.append(template.format(**dict(zip(keys, combo))))
    # dedupe while preserving order
    return list(dict.fromkeys(phrasings))


# Slot naming convention, so that expanded text stays grammatical:
#   Capitalised keys (Move, Place)  -> sentence-initial verbs
#   *_l keys (move_l, place_l)      -> the same verbs mid-sentence
#   to_target                       -> goes with motion verbs ("move it TO the target")
#   at_target                       -> goes with placement verbs ("place it AT the target")
TO_TARGET = ["to the target", "to the target position", "onto the goal region"]
AT_TARGET = ["on the target", "at the target position", "in the goal region"]

# Camera Alignment and Take Photo mark their target with a static cube rather than a
# goal region, so they get target wording that does not mention one.
TO_SPOT = ["to the target", "to the target position", "to the target spot"]
AT_SPOT = ["on the target", "at the target position", "on the target spot"]

TASKS = {
    "LiftBarrier": {
        "n_agents": 2,
        "templates": [
            "{Lift} the {obj} and keep it {level}.",
            "{Grasp} the {obj} {firmly} and {raise_l} it to the target height.",
            "Work together to {lift_l} the {obj} while keeping it {level}.",
            "{Lift} the {obj} evenly so that it stays {level}.",
            "{Grasp} the {obj} and {raise_l} it {smoothly} to the target height.",
            "Both arms {lift_l} the {obj}, keeping it {level} throughout.",
        ],
        "slots": {
            "Lift": ["Lift", "Raise", "Pick up", "Hoist"],
            "lift_l": ["lift", "raise", "pick up", "hoist"],
            "Grasp": ["Grasp", "Grip", "Take hold of"],
            "raise_l": ["raise", "lift", "bring"],
            "obj": ["metal barrier", "barrier", "steel barrier"],
            "level": ["straight", "level", "balanced", "horizontal"],
            "firmly": ["firmly", "securely", "with a stable grip"],
            "smoothly": ["smoothly", "steadily", "without tilting"],
        },
    },
    "PlaceFood": {
        "n_agents": 2,
        "templates": [
            "{Open} the {lid} and place {food} inside.",
            "{Open} the {lid} and move {food} to the {where} of the pot.",
            "{Open} the {lid}, then put {food} into the pot.",
            "One arm {opens} the {lid} while the other places {food} inside.",
            "{Open} the {lid} first, then transfer {food} into the pot.",
            "Hold the {lid} open and set {food} down in the {where} of the pot.",
            "{Open} the {lid} and {gently} set {food} in the {where} of the pot.",
        ],
        "slots": {
            "Open": ["Lift", "Open", "Raise"],
            "opens": ["lifts", "opens", "raises"],
            "lid": ["pot lid", "lid"],
            "food": ["a small piece of food", "the food item", "the meat", "the piece of meat"],
            "where": ["center", "middle"],
            "gently": ["carefully", "gently"],
        },
    },
    # Colours follow configs/table/two_robots_stack_cube.yaml (cubeA blue, cubeB green),
    # not the paper's Table IV, which says blue/red for this task.
    "TwoRobotsStackCube": {
        "n_agents": 2,
        "templates": [
            "{Move} the {blue} {to_target} and stack the {green} on top.",
            "{Place} the {blue} {at_target}, then stack the {green} on it.",
            "{Move} the {blue} {to_target} and {place_l} the {green} above it.",
            "First {move_l} the {blue} {to_target}, then stack the {green} on top.",
            "{Place} the {blue} {at_target} and stack the {green} on top of it.",
            "Position the {blue} {at_target} before stacking the {green}.",
        ],
        "slots": {
            "Move": ["Move", "Bring", "Transport"],
            "move_l": ["move", "bring", "carry"],
            "Place": ["Place", "Position", "Set"],
            "place_l": ["place", "position", "set"],
            "blue": ["blue cube", "blue block"],
            "green": ["green cube", "green block"],
            "to_target": TO_TARGET,
            "at_target": AT_TARGET,
        },
    },
    "CameraAlignment": {
        "n_agents": 3,
        "templates": [
            "{Place} the {obj} at its target position and {raise_l} the camera to match.",
            "{Hold} the {obj} at its target position and align the camera {precisely}.",
            "{Place} the {obj} {at_target}, then {align} the camera with it.",
            "{Move} the {obj} {to_target} and {raise_l} the camera to line up with it.",
            "{Hold} the {obj} steady {at_target} while the camera is {aligned}.",
            "{Place} the {obj} {at_target} and {align} the camera {precisely}.",
        ],
        "slots": {
            "obj": ["object", "item"],
            "Place": ["Place", "Position", "Set"],
            "Hold": ["Hold", "Keep", "Steady"],
            "Move": ["Move", "Bring", "Carry"],
            "raise_l": ["raise", "lift", "bring up"],
            "align": ["align", "line up", "match"],
            "aligned": ["aligned", "lined up", "brought into position"],
            "precisely": ["precisely", "carefully", "accurately"],
            "to_target": TO_SPOT,
            "at_target": AT_SPOT,
        },
    },
    # Colours follow configs/table/three_robots_stack_cube.yaml
    # (cubeA blue, cubeB green, cubeC red).
    "ThreeRobotsStackCube": {
        "n_agents": 3,
        "templates": [
            "{Place} the {blue}, then stack the {green} and {red} on top.",
            "{Place} the {blue} {at_target} and carefully stack the {green} and {red}.",
            "{Move} the {blue} {to_target}, then stack the {green} and the {red} on it.",
            "First {place_l} the {blue} {at_target}, then add the {green} and {red} on top.",
            "Position the {blue} {at_target} before stacking the {green} and {red}.",
            "{Place} the {blue} {at_target}, stack the {green}, then the {red}.",
        ],
        "slots": {
            "Place": ["Place", "Position", "Set"],
            "place_l": ["place", "position", "set"],
            "Move": ["Move", "Bring", "Carry"],
            "blue": ["blue cube", "blue block"],
            "green": ["green cube", "green block"],
            "red": ["red cube", "red block"],
            "to_target": TO_TARGET,
            "at_target": AT_TARGET,
        },
    },
    "TakePhoto": {
        "n_agents": 4,
        "templates": [
            "{Move} the {obj} {to_target}, {align} the camera, then press the shutter.",
            "{Place} the {obj} {at_target}, {align} the camera, and press the shutter {after}.",
            "{Move} the {obj} {to_target}, {align} the camera with it, then {press} the shutter.",
            "{Place} the {obj} {at_target} and {align} the camera before {pressing} the shutter.",
            "First {move_l} the {obj} {to_target}, then {align} the camera and {press} the shutter.",
            "{Place} the {obj} {at_target}, line the camera up, and {press} the shutter.",
        ],
        "slots": {
            "obj": ["object", "item"],
            "Move": ["Move", "Bring", "Transport"],
            "move_l": ["move", "bring", "carry"],
            "Place": ["Place", "Position", "Set"],
            "align": ["align", "line up", "aim"],
            "press": ["press", "push", "trigger"],
            "pressing": ["pressing", "pushing", "triggering"],
            "after": ["afterward", "last", "at the end"],
            "to_target": TO_SPOT,
            "at_target": AT_SPOT,
        },
    },
}


def generate(task_name, seed=0):
    spec = TASKS[task_name]
    phrasings = _expand(spec["templates"], spec["slots"])

    needed = N_TRAIN + N_EVAL
    if len(phrasings) < needed:
        raise RuntimeError(
            f"{task_name}: only {len(phrasings)} unique phrasings, need {needed}. "
            "Add more templates or slot values."
        )

    rng = random.Random(seed)
    rng.shuffle(phrasings)

    return {
        "task_name": task_name,
        "n_agents": spec["n_agents"],
        "n_unique_phrasings": len(phrasings),
        "train": sorted(phrasings[:N_TRAIN]),
        "eval": sorted(phrasings[N_TRAIN:needed]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", type=str, default=None, help="Task to generate for")
    parser.add_argument("--all", action="store_true", help="Generate for every known task")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.all and args.task_name is None:
        parser.error("pass --task_name or --all")

    task_names = list(TASKS) if args.all else [args.task_name]
    os.makedirs(args.out_dir, exist_ok=True)

    for task_name in task_names:
        if task_name not in TASKS:
            raise KeyError(f"unknown task {task_name}; known: {sorted(TASKS)}")
        bank = generate(task_name, seed=args.seed)
        out_path = os.path.join(args.out_dir, f"{task_name}.json")
        with open(out_path, "w") as f:
            json.dump(bank, f, indent=2)
        print(
            f"{task_name}: {bank['n_unique_phrasings']} unique phrasings -> "
            f"{len(bank['train'])} train / {len(bank['eval'])} eval -> {out_path}"
        )


if __name__ == "__main__":
    main()
