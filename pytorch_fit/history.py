def add_to_history(history_dct, item_dct):
    for key, value in item_dct.items():
        if key not in history_dct:
            history_dct[key] = value
        else:
            if type(value) is dict:
                add_to_history(history_dct[key], value)
            else:
                history_dct[key] += value
