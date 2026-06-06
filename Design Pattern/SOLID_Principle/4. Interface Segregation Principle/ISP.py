from abc import ABC, abstractmethod


# =====================================================
# Interface 1: Work capability
# =====================================================
class Workable(ABC):

    @abstractmethod
    def work(self):
        pass


# =====================================================
# Interface 2: Eat capability
# =====================================================
class Eatable(ABC):

    @abstractmethod
    def eat(self):
        pass


# =====================================================
# Interface 3: Sleep capability
# =====================================================
class Sleepable(ABC):

    @abstractmethod
    def sleep(self):
        pass


# =====================================================
# Human can work, eat and sleep
# =====================================================
class Human(Workable, Eatable, Sleepable):

    def work(self):
        print("Human is working")

    def eat(self):
        print("Human is eating lunch")

    def sleep(self):
        print("Human is sleeping")


# =====================================================
# Robot only works
# No need to implement eat() or sleep()
# =====================================================
class Robot(Workable):

    def work(self):
        print("Robot is assembling products")


# =====================================================
# Dog can eat and sleep but doesn't work
# =====================================================
class Dog(Eatable, Sleepable):

    def eat(self):
        print("Dog is eating food")

    def sleep(self):
        print("Dog is sleeping")


# =====================================================
# Client Functions
# =====================================================

def assign_work(worker: Workable):
    worker.work()


def feed(entity: Eatable):
    entity.eat()


def make_sleep(entity: Sleepable):
    entity.sleep()


# =====================================================
# Main
# =====================================================

human = Human()
robot = Robot()
dog = Dog()

print("----- Working -----")
assign_work(human)
assign_work(robot)

print("\n----- Eating -----")
feed(human)
feed(dog)

print("\n----- Sleeping -----")
make_sleep(human)
make_sleep(dog)