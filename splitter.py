import random

def train_val_splitter(x, y, val_split=0.25):
    """Splits x and y iterables randomly using the val_split value.

    Args:
        x (_type_): _description_
        y (_type_): _description_
        val_split (float, optional): _description_. Defaults to 0.25.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    x_val = []
    y_val = []

    x_length = len(x)
    y_length = len(y)
    val_count = int(x_length * val_split)

    if x_length != y_length:
        raise Exception(f"ERROR | Input length mismatch | x_len = {x_length}, y_len = {y_length}")
    
    validationIndexes = random.sample(range(0, x_length), val_count)

    x_val = [x[i] for i in validationIndexes]
    x = [i for j, i in enumerate(x) if j not in validationIndexes]
    y_val = [y[i] for i in validationIndexes]
    y = [i for j, i in enumerate(y) if j not in validationIndexes]
    return x, x_val, y, y_val