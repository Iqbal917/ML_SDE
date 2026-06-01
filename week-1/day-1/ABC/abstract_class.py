from abc import ABC, abstractmethod

class PaymentProcess(ABC):
    @abstractmethod
    def process_payment(self, amount):
        pass 

class CreditCardProcessor(PaymentProcess):
    def process_payment(self, amount):
        print(f"Processing credit card payment of ${amount}")

processor = CreditCardProcessor()
processor.process_payment(100)

