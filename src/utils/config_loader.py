import yaml
from pathlib import Path
import os
import time

BASE_DIR = Path(__file__).resolve().parents[2]  # points to /projects
CONFIG_DIR = Path(os.getenv("APP_CONFIG_DIR", BASE_DIR / "config"))

class _CachedYaml:
    def __init__(self, filename: str):
        self.path = CONFIG_DIR / filename
        self.mtime = 0.0
        self.value = None

    def get(self):
        mt = self.path.stat().st_mtime
        if self.value is None or mt != self.mtime:
            with open(self.path, "r", encoding="utf-8") as f:
                self.value = yaml.safe_load(f) or {}
            self.mtime = mt
        return self.value

categories_cache = _CachedYaml("categories.yaml")
settings_cache   = _CachedYaml("settings.yaml")

def load_categories() -> dict:
    return categories_cache.get()

def load_settings() -> dict:
    return settings_cache.get()


if __name__ == "__main__":
    cats = load_categories()
    settings = load_settings()
    print(cats)