class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello my name is {self.name} & my age is {self.age}")

p1 = Person("John", 38)
p1.greet()

class Car:
    def __init__(self, brand):
        self.brand = brand
    
    def show(self):
        print(self.brand)

c1 = Car("Ford")
c1.show()

class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

s1 = Student("Anna", "A")
print(s1.grade)
s1.grade = "B"
print(s1.grade)

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

r1 = Rectangle(5,3)
print(r1.area())

class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(self.name)

class Dog(Animal):
    pass 

d1 = Dog("Rex")
d1.speak()

class Cat:
    def sound(self):
        print("Meow")

class Fox:
    def sound(self):
        print("Wa-pa-pa-pa-pa-pow!")

c1 = Cat()
f1 = Fox()
c1.sound()
f1.sound()

class ScoreBoard:
    def __init__(self, score):
        self.__score = score

    def get_score(self):
        return self.__score

s1 = ScoreBoard(0)
print(s1.get_score())


