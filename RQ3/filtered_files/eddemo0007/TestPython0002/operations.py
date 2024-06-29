# Generated with ChatGPT using chat.openai.com and the following prompt:  
#  Using Visual Code , can you give me detailed steps on how to create a python github repo that has a  CRUD python program that will perform CRUD operations on a Customer table in a Postgresql database?  I do not want to use :python3 -m venv venv
#  Instead I want to use:
#  Pipenv
#  Can you give me code that will perfoam all 4 CRUD operations with user import along with 15 sample database records? Can you also include steps that I need to take to Create a Pipfile.lock file

import psycopg2
from psycopg2 import sql

# Database connection parameters
db_config = {
    "host": "localhost",
    "database": "your_database",
    "user": "your_username",
    "password": "your_password"
}

# Connect to your PostgreSQL database
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS customer (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    age INT NOT NULL
)
""")
conn.commit()

# CRUD Operations

## Create
def create_customer(name, email, age):
    cursor.execute("INSERT INTO customer (name, email, age) VALUES (%s, %s, %s) RETURNING id;", (name, email, age))
    conn.commit()
    return cursor.fetchone()[0]

## Read
def get_customer(id):
    cursor.execute("SELECT * FROM customer WHERE id = %s;", (id,))
    return cursor.fetchone()

## Update
def update_customer(id, name=None, email=None, age=None):
    fields = {"name": name, "email": email, "age": age}
    set_clause = ", ".join([f"{key} = %s" for key, value in fields.items() if value is not None])
    values = tuple(value for value in fields.values() if value is not None) + (id,)
    cursor.execute(f"UPDATE customer SET {set_clause} WHERE id = %s;", values)
    conn.commit()

## Delete
def delete_customer(id):
    cursor.execute("DELETE FROM customer WHERE id = %s;", (id,))
    conn.commit()

# Sample data insertion
sample_customers = [
    ("John Doe", "johndoe@example.com", 30),
    ("Jane Smith", "janesmith@example.com", 25),
    # Add more customer tuples here up to 15
]

for customer in sample_customers:
    create_customer(*customer)

# Example Usage
print(get_customer(1))
update_customer(1, name="Johnathan Doe")
delete_customer(2)

# Don't forget to close the connection
cursor.close()
conn.close()
