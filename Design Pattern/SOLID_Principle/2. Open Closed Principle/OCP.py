from abc import ABC, abstractmethod

# =====================================================
# Abstract Payment Contract
#
# Any payment method must implement pay().
# Checkout depends on this abstraction.
# =====================================================
class PaymentMethod(ABC):

    @abstractmethod
    def pay(self, amount):
        pass


# =====================================================
# Payment Method #1
# =====================================================
class CreditCardPayment(PaymentMethod):

    def pay(self, amount):
        print(f"Paid ₹{amount} using Credit Card")


# =====================================================
# Payment Method #2
# =====================================================
class UPIPayment(PaymentMethod):

    def pay(self, amount):
        print(f"Paid ₹{amount} using UPI")


# =====================================================
# Payment Method #3
#
# Added later without modifying Checkout.
# This demonstrates OCP.
# =====================================================
class PayPalPayment(PaymentMethod):

    def pay(self, amount):
        print(f"Paid ₹{amount} using PayPal")


# =====================================================
# Payment Method #4
#
# Another extension
# =====================================================
class NetBankingPayment(PaymentMethod):

    def pay(self, amount):
        print(f"Paid ₹{amount} using Net Banking")


# =====================================================
# Checkout Service
#
# Closed for modification.
# It never changes when new payment methods are added.
# =====================================================
class Checkout:

    def process_payment(
        self,
        payment_method: PaymentMethod,
        amount
    ):
        payment_method.pay(amount)


# =====================================================
# Client Code
# =====================================================

checkout = Checkout()

checkout.process_payment(
    CreditCardPayment(),
    1000
)

checkout.process_payment(
    UPIPayment(),
    500
)

checkout.process_payment(
    PayPalPayment(),
    750
)

checkout.process_payment(
    NetBankingPayment(),
    2000
)