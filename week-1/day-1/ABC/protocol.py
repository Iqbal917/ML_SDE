from typing import Protocol

class PayementProcessor(Protocol):
    def process_payment(self, amount:float) -> None:
        ...

class CreditCardProcessor(PayementProcessor):
    def process_payment(self, amount:float) -> None:
        print(f"Processing ${amount}")

def checkout(processor: PayementProcessor):
    processor.process_payment(50)

card = CreditCardProcessor()
checkout(card)

#--------------------------------------------------------------------------------------

class Logger(Protocol):
    def log(self, message: str) -> None:
        ...


class ConsoleLogger:
    def log(self, message: str) -> None:
        print(message)


class FileLogger:
    def log(self, message: str) -> None:
        with open("app.log", "a") as f:
            f.write(message + "\n")


def run(logger: Logger):
    logger.log("Application started")


run(ConsoleLogger())
run(FileLogger())
