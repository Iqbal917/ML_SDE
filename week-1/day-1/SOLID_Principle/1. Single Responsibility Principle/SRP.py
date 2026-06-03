# =====================================================
# Item: Responsible only for storing item information
# =====================================================
class Item:

    def __init__(self, name, price):
        self.name = name
        self.price = price


# =====================================================
# Order: Responsible only for storing order data
# =====================================================
class Order:

    def __init__(self, order_id, items):
        self.order_id = order_id
        self.items = items


# =====================================================
# PricingService:
# Responsible only for pricing calculations
# If tax/discount logic changes, only this class changes.
# =====================================================
class PricingService:

    def calculate_total(self, order):
        return sum(item.price for item in order.items)


# =====================================================
# OrderManager:
# Responsible only for processing/placing orders
# =====================================================
class OrderManager:

    def __init__(self, pricing_service):
        self.pricing_service = pricing_service

    def place_order(self, order):

        total = self.pricing_service.calculate_total(order)

        print("Order placed successfully")
        print(f"Order ID : {order.order_id}")
        print(f"Total    : ${total:.2f}")


# =====================================================
# Client Code
# =====================================================

items = [
    Item("Book", 12.99),
    Item("Pen", 1.99),
    Item("Notebook", 5.49)
]

order = Order(101, items)

pricing_service = PricingService()

order_manager = OrderManager(pricing_service)

order_manager.place_order(order)