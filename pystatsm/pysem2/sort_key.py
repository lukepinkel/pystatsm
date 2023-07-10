
import re


def _default_sort_key(item):
    match = re.match(r"([a-zA-Z]+)(\d+)", item)
    if match:
        alphabetic_part = match.group(1)
        numeric_part = int(match.group(2))
    else:
        alphabetic_part = item
        numeric_part = 0
    return (alphabetic_part, numeric_part)
