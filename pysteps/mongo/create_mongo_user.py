import secrets
import string
import os
from pymongo import MongoClient
from urllib.parse import quote_plus

# === CONFIGURATION ===
MONGO_HOST = "localhost"
MONGO_PORT = 27017
AUTH_DB = "admin"
MONGO_ADMIN_USER = os.getenv("MONGO_USER")
MONGO_ADMIN_PASS = os.getenv("MONGO_PWD")
TARGET_DB = "STEPS"
PWD_DEFAULT = "c-bandBox" 

# === FUNCTIONS ===
def generate_password(length=16):
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def create_user(username, role="readWrite"):
    # password = generate_password()
    password = PWD_DEFAULT 
    client = MongoClient(f"mongodb://{quote_plus(MONGO_ADMIN_USER)}:{quote_plus(MONGO_ADMIN_PASS)}@{MONGO_HOST}:{MONGO_PORT}/?authSource={AUTH_DB}")
    db = client[TARGET_DB]

    try:
        db.command("createUser", username, pwd=password, roles=[{"role": role, "db": TARGET_DB}])
        print(f"\n✅ User '{username}' created with role '{role}'.\n")
        print("Connection string:")
        print(f"  mongodb://{quote_plus(username)}:{quote_plus(password)}@{MONGO_HOST}:{MONGO_PORT}/{TARGET_DB}?authSource={TARGET_DB}\n")
    except Exception as e:
        print(f"❌ Failed to create user '{username}': {e}")

# === ENTRY POINT ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a MongoDB user with a random password.")
    parser.add_argument("username", help="Username to create")
    parser.add_argument("--role", default="readWrite", help="MongoDB role (default: readWrite)")
    args = parser.parse_args()
    create_user(args.username, args.role)

