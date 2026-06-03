from abc import ABC, abstractmethod 

#Common Interface
class Notification(ABC):
    @abstractmethod
    def send(self, message):
        pass 

# Concrete class
class EmailNotification(Notification):
    def send(self, message):
        print(f"Send Email: {message}")

class SMSlNotification(Notification):
    def send(self, message):
        print(f"Send SMS: {message}")

class PUSHNotification(Notification):
    def send(self, message):
        print(f"Send PUSH Notification: {message}")

#Factory 
class NotificationFactory:
    @staticmethod 
    def create_notification(notification_type):
        if notification_type == "email":
            return EmailNotification()
        
        elif notification_type == "SMS":
            return SMSlNotification()
        
        elif notification_type == "PUSH":
            return PUSHNotification()

        raise ValueError(f"Unknown notification typer: {notification_type}")

# Client Conde 
notification = NotificationFactory.create_notification("email")
notification.send("Welcome")

notification = NotificationFactory.create_notification("SMS")
notification.send("OTP: 1234")

notification = NotificationFactory.create_notification("PUSH")
notification.send("New Message Recieved")
