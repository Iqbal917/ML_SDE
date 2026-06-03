with open("demofile.txt", "w") as f:
    f.write("Now the file has content")

with open("demofile.txt") as f:
    print(f.read())

with open("demofile.txt", "a") as f:
    f.write("second line")

with open("demofile.txt") as f:
    print(f.read())
