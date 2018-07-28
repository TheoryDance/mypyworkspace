class Sample:
    def __enter__(self):
        print("In __enter__()")
        return "Foo"

    def __exit__(self, type, value, trace):
        print("In __exit__()")

    def myname(self):
        print('this is Sample instanse.')


def get_sample():
    return Sample()


with get_sample() as sample:
    print(type(get_sample()))
    print(type(sample))
    print("sample: ", sample)
    print('hello')