"""
Pre-processing and text cleaning utilities.
"""

def remove_illegal_chars(text: str) -> str:
    """
    Removes illegal characters from the given text.

    Args:
        text (str): The text from which illegal characters need to be removed.

    Returns:
        str: The text with illegal characters removed.
    """
    illegal_chars = [
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06',
        '\x07', '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10',
        '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17',
        '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e',
        '\x1f'
    ]
    for char in illegal_chars:
        text = text.replace(char, '')

    return text