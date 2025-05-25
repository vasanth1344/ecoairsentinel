import sqlite3

DB_NAME = "new.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS node1_sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,
            humidity REAL,
            pm25 REAL,
            pm10 REAL,
            timestamp TEXT
        );
    ''')

    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS node2_sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,
            humidity REAL,
            pm25 REAL,
            pm10 REAL,
            timestamp TEXT
        );
    ''')

    

    conn.commit()
    conn.close()

def insert_node1_data(temperature, humidity, pm25, pm10, timestamp):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(''' 
        INSERT INTO node1_sensor_readings (temperature, humidity, pm25, pm10, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (temperature, humidity, pm25, pm10, timestamp))
    conn.commit()
    conn.close()

def insert_node2_data(temperature, humidity, pm25, pm10, timestamp):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(''' 
        INSERT INTO node2_sensor_readings (temperature, humidity, pm25, pm10, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (temperature, humidity, pm25, pm10, timestamp))
    conn.commit()
    conn.close()