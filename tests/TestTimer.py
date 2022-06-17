from picutils import PICTimer

if __name__ == '__main__':
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