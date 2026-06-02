from abc import ABC, abstractmethod

# Interface
class Notification(ABC):
    @abstractmethod
    def send(self, message: str) -> None:
        pass

# Concrete Class 
class EmailNotification(Notification):
    def send(self, message: str) -> None:
        print(f"Email: {message}")


class SMSNotification(Notification):
    def send(self, message: str) -> None:
        print(f"SMS: {message}")


class PushNotification(Notification):
    def send(self, message: str) -> None:
        print(f"PUSH: {message}")

# Factory 
class NotificationFactory:
    _registry: dict[str, type[Notification]] = {
        "email": EmailNotification,
        "sms": SMSNotification,
        "push": PushNotification,
    }

    @classmethod
    def create(cls, notification_type: str) -> Notification:
        try:
            return cls._registry[notification_type]()
        except KeyError:
            raise ValueError(
                f"Unsupported notification type: {notification_type}"
            )

notification = NotificationFactory.create("email")
notification.send("Hello")
