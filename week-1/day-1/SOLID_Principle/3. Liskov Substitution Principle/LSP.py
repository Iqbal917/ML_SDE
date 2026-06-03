from abc import ABC, abstractmethod


# =====================================================
# Base Class
# =====================================================
class Bird:
    pass


# =====================================================
# Specialized Bird Type
# Only birds that can fly should inherit from this
# =====================================================
class FlyingBird(Bird, ABC):

    @abstractmethod
    def fly(self):
        pass


# =====================================================
# Flying Birds
# =====================================================
class Sparrow(FlyingBird):

    def fly(self):
        print("Sparrow is flying")


class Eagle(FlyingBird):

    def fly(self):
        print("Eagle is soaring high")


class Parrot(FlyingBird):

    def fly(self):
        print("Parrot is flying")


# =====================================================
# Non-Flying Bird
# =====================================================
class Ostrich(Bird):

    def run(self):
        print("Ostrich is running")


# =====================================================
# Client Function
# Depends on FlyingBird abstraction
# =====================================================
def make_bird_fly(bird: FlyingBird):
    """
    This function expects any FlyingBird.

    According to LSP:
    Any subclass of FlyingBird should work here
    without changing the behavior.
    """
    bird.fly()


# =====================================================
# Main
# =====================================================
sparrow = Sparrow()
eagle = Eagle()
parrot = Parrot()

make_bird_fly(sparrow)
make_bird_fly(eagle)
make_bird_fly(parrot)

print()

ostrich = Ostrich()
ostrich.run()