# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    ls = ['hello', 'world']
    ls.append('python')
    print(ls)

    import time
    from datetime import datetime

    for i in range(10):
        now = datetime.now()
        current_time = now.strftime("%H %M %S")
        print(current_time)
        time.sleep(10)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
