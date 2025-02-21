import itertools

def create_deck():
    return [i for i in range(2, 11)] + list(itertools.cycle(['T', 'J', 'Q', 'K', 'A']))
