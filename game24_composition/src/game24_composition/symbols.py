"""Fixed symbol mapping for the composition sandbox."""

SYMBOL_TO_NUMBER = {
    "dax": 1,
    "mip": 2,
    "zup": 3,
    "wug": 4,
    "tav": 5,
    "lorn": 6,
    "pesh": 7,
    "nari": 8,
    "feg": 9,
    "somb": 10,
}

NUMBER_TO_SYMBOL = {value: symbol for symbol, value in SYMBOL_TO_NUMBER.items()}
SYMBOLS = tuple(SYMBOL_TO_NUMBER)


def symbols_to_numbers(symbols, mapping=None):
    mapping = mapping or SYMBOL_TO_NUMBER
    return [mapping[symbol] for symbol in symbols]


def numbers_to_symbols(numbers):
    return [NUMBER_TO_SYMBOL[int(number)] for number in numbers]
