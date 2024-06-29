from numbers import Number
from enum import Enum
import json
from collections import defaultdict
import tldextract


def preprocess_bytes(data):
    data = data.strip(b"\x00")
    try:
        # Attempt to decode bytes to a UTF-8 string
        data = data.decode('utf-8')
    except UnicodeDecodeError:
        # Return hexadecimal representation if decoding fails
        data = data.hex()
    return data


def preprocess_numbers(data):
    try:
        # Attempt conversion to int if it matches exactly
        data = int(data) if data == int(data) else float(data)
    except (ValueError, TypeError, OverflowError, NotImplementedError):
        # If conversion to int fails, fall back to float
        data = float(data)
    return data


def preprocess_data(data):
    """Generated with ChatGPT.

    Recursively process nested data: decode bytes to UTF-8 strings, convert numbers to integers or
    floats.
    """

    if isinstance(data, (Enum, bool, type(None))):
        return data
    if isinstance(data, dict):
        # Recursively apply transformations to each key-value pair in the dictionary
        data = {preprocess_data(key): preprocess_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Apply transformations to each item in the list
        data = [preprocess_data(item) for item in data]
    elif isinstance(data, bytes):
        data = preprocess_bytes(data)
    elif isinstance(data, Number):
        data = preprocess_numbers(data)
    if not isinstance(data, (dict, list, str, int, float)):
        try:
            # If the data type is not in the allowed types, convert it to a string
            data = str(data)
        except Exception as e:
            raise ValueError(f"Unsupported data type: {type(data)}: {repr(data)}") from e

    return data


def extract_2ld(fqdn):
    """Generated with ChatGPT.

    Extract the second level domain from a fully qualified domain name (FQDN).
    :param fqdn: Fully qualified domain name
    :return: 2LD and TLD combined
    """

    if not fqdn:
        return ""

    # First try with tldextract
    extracted = tldextract.extract(fqdn)
    if extracted.domain and extracted.suffix:
        # If both parts are identified, return them
        return f"{extracted.domain}.{extracted.suffix}"

    # Fallback mechanism
    parts = fqdn.split('.')

    # Basic assumption: The last two parts are the domain and TLD
    if len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"

    # Return the original
    return fqdn


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.name
        return super().default(o)

    def tuple_to_string(self, value):
        if isinstance(value, tuple):
            return f"({", ".join(str(k) for k in value)})"
        return value

    def encode(self, o):
        if isinstance(o, dict):
            # Convert tuple keys to strings
            return super().encode({self.tuple_to_string(k): v for k, v in o.items()})
        return super().encode(o)


def collect_keys(data: list):
    dict_keys = set()
    # Collect all unique keys from each dictionary in the list
    for item in data:
        if isinstance(item, dict):
            dict_keys.update(item.keys())
    return dict_keys


def append_value_if_any(source_dict, dict_keys, target_dict, prefix, sep):
    for dkey in dict_keys:
        target_dict[f"{prefix}{sep}{dkey}"].append(source_dict.get(dkey, None))


def process_item(items, sep, key, value, prefix=''):
    # Construct the full key from prefix and current key with proper handling for nested lists
    full_key = f"{prefix}{sep}{key}" if prefix else key

    if isinstance(value, dict):
        # Recurse into dictionaries
        for sub_key, sub_val in flatten_dict(value, full_key, sep).items():
            items.append((sub_key, sub_val))
    elif isinstance(value, list):
        # Handle list, may contain dicts or other lists
        list_items = defaultdict(list)
        simple_values = []
        dict_keys = collect_keys(value)

        for i, item in enumerate(value):
            if isinstance(item, dict):
                append_value_if_any(item, dict_keys, list_items, full_key, sep)
            elif isinstance(item, list):
                # Recursive call to handle nested lists with correct indexing
                process_item(items, sep, i, item, full_key)
            else:
                simple_values.append(item)

        for list_key, list_vals in list_items.items():
            items.append((list_key, list_vals))
        if simple_values:
            items.append((full_key, simple_values))
    else:
        items.append((full_key, value))


def flatten_dict(d, parent_key='', sep='.'):
    """Generated with ChatGPT."""
    items = []
    for key, val in d.items():
        process_item(items, sep, key, val, parent_key)
    return dict(items)


def convert_mac(mac):
    """Generated with ChatGPT.

    Convert a MAC address to a colon-separated hexadecimal string.
    """
    return ":".join([f"{int(byte):02x}" for byte in mac.strip(b"\x00")])
