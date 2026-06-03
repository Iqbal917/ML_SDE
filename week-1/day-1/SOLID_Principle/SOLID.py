from abc import ABC, abstractmethod


# =====================================================
# SRP (Single Responsibility Principle)
# =====================================================
# Order class is responsible ONLY for storing order data.
# It does not handle payments, notifications, databases, etc.
# =====================================================

class Order:
    def __init__(self, order_id, amount):
        self.order_id = order_id
        self.amount = amount


# =====================================================
# OCP (Open/Closed Principle)
# =====================================================
# New payment methods can be added by creating a new class
# without modifying existing checkout code.
# =====================================================

class PaymentMethod(ABC):

    @abstractmethod
    def pay(self, amount):
        pass


class CreditCardPayment(PaymentMethod):

    def pay(self, amount):
        print(f"Paid ₹{amount} using Credit Card")


class UPIPayment(PaymentMethod):

    def pay(self, amount):
        print(f"Paid ₹{amount} using UPI")


# Adding a new payment method later requires NO changes
# to CheckoutService.
class PayPalPayment(PaymentMethod):

    def pay(self, amount):
        print(f"Paid ₹{amount} using PayPal")


# =====================================================
# ISP (Interface Segregation Principle)
# =====================================================
# Instead of forcing all notification providers
# to implement unnecessary methods, we create
# a focused interface.
# =====================================================

class NotificationService(ABC):

    @abstractmethod
    def send(self, message):
        pass


class EmailNotification(NotificationService):

    def send(self, message):
        print(f"EMAIL: {message}")


class SMSNotification(NotificationService):

    def send(self, message):
        print(f"SMS: {message}")


# =====================================================
# DIP (Dependency Inversion Principle)
# =====================================================
# High-level business logic should depend on abstractions
# (PaymentMethod, NotificationService)
# instead of concrete implementations.
# =====================================================

class CheckoutService:

    def __init__(
        self,
        payment_method: PaymentMethod,
        notification_service: NotificationService
    ):
        # Inject dependencies through constructor
        self.payment_method = payment_method
        self.notification_service = notification_service

    def checkout(self, order: Order):

        # Process payment
        self.payment_method.pay(order.amount)

        # Send notification
        self.notification_service.send(
            f"Order {order.order_id} placed successfully."
        )


# =====================================================
# LSP (Liskov Substitution Principle)
# =====================================================
# Any subclass of PaymentMethod can replace another
# without breaking CheckoutService.
#
# CreditCardPayment
# UPIPayment
# PayPalPayment
#
# CheckoutService doesn't care which one it gets.
# =====================================================


# =====================================================
# APPLICATION CODE
# =====================================================

order = Order(
    order_id="ORD1001",
    amount=2500
)

# Dependency Injection
payment = UPIPayment()
notification = EmailNotification()

checkout_service = CheckoutService(
    payment_method=payment,
    notification_service=notification
)

checkout_service.checkout(order)