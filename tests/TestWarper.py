class Test:
    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, func):
        def InnerWarper():
            return lambda x: x - 1
        return InnerWarper

@Test('a')
def f1():
    print('f1')

if __name__ == '__main__':
    a = f1()
    print(a)