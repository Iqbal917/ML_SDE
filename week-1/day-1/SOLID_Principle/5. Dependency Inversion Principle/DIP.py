from abc import ABC, abstractmethod


# =====================================================
# Abstraction (High-level modules depend on this)
# =====================================================
class Database(ABC):

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def fetch_users(self):
        pass


# =====================================================
# Low-level implementation #1
# =====================================================
class MySQLDatabase(Database):

    def connect(self):
        print("Connected to MySQL Database")

    def fetch_users(self):
        return [
            {"id": 1, "name": "John"},
            {"id": 2, "name": "Alice"}
        ]


# =====================================================
# Low-level implementation #2
# =====================================================
class PostgreSQLDatabase(Database):

    def connect(self):
        print("Connected to PostgreSQL Database")

    def fetch_users(self):
        return [
            {"id": 101, "name": "Bob"},
            {"id": 102, "name": "Emma"}
        ]


# =====================================================
# High-level module
# Depends on abstraction, not implementation
# =====================================================
class UserService:

    def __init__(self, database: Database):
        self.database = database

    def get_users(self):
        self.database.connect()

        users = self.database.fetch_users()

        print("\nUsers:")
        for user in users:
            print(f"ID: {user['id']}, Name: {user['name']}")

        return users


# =====================================================
# Client Code
# =====================================================

print("Using MySQL\n")

mysql_db = MySQLDatabase()
user_service = UserService(mysql_db)

user_service.get_users()

print("\n" + "=" * 40 + "\n")

print("Using PostgreSQL\n")

postgres_db = PostgreSQLDatabase()
user_service = UserService(postgres_db)

user_service.get_users()