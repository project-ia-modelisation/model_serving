def greet(name):
    return f"Hello serving, {name}!"

if __name__ == "__main__":
    name = input("Enter your name: ")
    print(greet(name))