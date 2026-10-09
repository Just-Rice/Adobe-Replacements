
# TODO: De-duplicate this
def print_title(title, length=40):
    print(f'\n{"/"*length}')
    print(title.center(length, ' '))
    print(f'{"/"*length}\n', flush=True)
