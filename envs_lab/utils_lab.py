def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if isinstance(val, list):
            element = [class_to_dict(item) for item in val]
        else:
            element = class_to_dict(val)
        result[key] = element
    return result
