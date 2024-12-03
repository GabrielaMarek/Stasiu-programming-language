import Stasiu

while True:
    text = input('Stasiu >')
    result, error = Stasiu.run('<stdin>', text)

    if error: print(error.stringify())
    else: print(result)
