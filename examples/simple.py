def a_gin_configurable_fn(print_yes):
    if print_yes:
        print("Yes")
    else:
        print("No")


def simple_program(quinfig):
    print("Running simple program.")
    print(quinfig)
    print("Pay attention to what's being printed below.")
    a_gin_configurable_fn()

