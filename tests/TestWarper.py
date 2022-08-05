from picutils.RecursiveWarper import make_recursive_func

class Test:
    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, func):
        def InnerWarper():
            return lambda x: x - 1
        return InnerWarper

# @Test('a')
@make_recursive_func
def f1(x):
    print(x)

if __name__ == '__main__':
    a = f1()
    print(a)