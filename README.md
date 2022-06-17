# PIC DIBR utils

## install
~~~
pip install https://github.com/picpic2013/picutils/releases/download/v0.0.1/picutils-0.0.1.tar.gz
~~~

## usage

### PIC_Timer
~~~ python
# example 1
# function timer
print('='*50, 'example 1', '='*50)
@PICTimer
def f1():
    for _ in range(100000):
        b = _

f1()

# example 2
# with code block
print('='*50, 'example 2', '='*50)
with PICTimer.getTimer('example2') as t:
    for _ in range(1, 100000):
        if _ % 20000 == 0:
            t.showTime()
        bb = _

# example 3
print('='*50, 'example 3', '='*50)

timer = PICTimer.getTimer('example3') 
timer.startTimer()                    
# or `timer = PICTimer.getTimer('example3', autoStart=True)`

for idx in range(3):
    for _ in range(1, 100000):
        if _ % 20000 == 0:
            timer.showTime("stage_" + str(_))
        bb = _
    timer.forceShowTime() # you can output results before summary
timer.summary()

# example 4
# create sub-timer
print('='*50, 'example 4', '='*50)
timer = PICTimer.getTimer('example4', autoStart=True)
for idx in range(3):
    subTimer = timer.getTimer('sub_' + str(idx), autoStart=True)
    for _ in range(1, 100000):
        if _ % 20000 == 0:
            subTimer.showTime("stage_" + str(_))
        bb = _
    timer.showTime()
timer.summary()
~~~

### make_recursive_func
~~~ python
# dict args
arg1 = {idx: 'arg1_k_'+str(idx) for idx in range(3)}
arg2 = {idx: 'arg2_k_'+str(idx) for idx in range(3)}

# list args
arg1 = [_ for _ in range(10)]
arg2 = [_ for _ in range(10)]

# tuple / generator args
arg1 = (_ for _ in range(10))
arg2 = (_ for _ in range(10))

# non-iterable args
arg1 = 0
arg2 = 1

# mutiple args
arg1 = {idx: ['arg1_k_'+str(idx)+'_l_'+str(_) for _ in range(2)] for idx in range(3)}
arg2 = {idx: ['arg2_k_'+str(idx)+'_l_'+str(_) for _ in range(2)] for idx in range(3)}

@make_recursive_func
def f1(x, base):
    return base.format(x)

@make_recursive_func
def f2(x, y, base):
    return base.format(x, y)

@make_recursive_func
def f3(x, y, base1, base2):
    return base1.format(x), base2.format(y)

@make_multi_return_recursive_func
def f4(x, y, base1, base2):
    return base1.format(x), base2.format(y)

res1 = f1(arg1, base="arg1: {}")
res2 = f2(arg1, arg2, base="arg1: {}, arg2: {}")
res3 = f3(arg1, arg2, base1="arg1: {}", base2=" arg2: {}")
res4 = f4(arg1, arg2, base1="arg1: {}", base2=" arg2: {}")

print(res1)
print(res2)
print(res3)
print(res4)
~~~