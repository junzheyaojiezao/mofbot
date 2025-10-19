import json, os
from typing import List, Dict

class Memory:
    def __init__(self, path: str = "./memory/state.jsonl", max_turns: int = 8):
        self.path = path
        self.max_turns = max_turns
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.buffer: List[Dict[str, str]] = []

    def add_turn(self, user: str, assistant: str):
        self.buffer.append({"user": user, "assistant": assistant})
        self.buffer = self.buffer[-self.max_turns:]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.buffer[-1], ensure_ascii=False) + "\n")


    def load_recent(self):
        return self.buffer
