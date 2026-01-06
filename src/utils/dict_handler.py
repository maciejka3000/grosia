from typing import List, Dict, Any

def merge_dicts(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {}
    for d in list_of_dicts:
        merged.update(d)
    return merged