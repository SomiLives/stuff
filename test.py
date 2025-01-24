
import mysql.connector
import logging
import os

logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': os.getenv('DB_PASSWORD', 'somnotho'),
    'database': 'LargeECommerceDB',
}

def get_db_connection():
    """Establish a database connection."""
    try:
        logger.info("Attempting to connect to the database with config: %s", DB_CONFIG)
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            logger.info("Successfully connected to the database.")
            return connection
        else:
            logger.error("Failed to connect to the database.")
            return None
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to MySQL: {err}")
        return None

if __name__ == "__main__":
    connection = get_db_connection()
    if connection:
        print("Connection successful!")
    else:
        print("Connection failed.")
