import numpy as np
import yfinance as yf
import mysql.connector
from datetime import datetime

# Connessione al database
mydb = mysql.connector.connect(
    host="localhost",  
    user="root",  
    password="",  
    database="VaR_database"  
)

cursor = mydb.cursor() 

# Creazione delle tabelle se non esistono
cursor.execute('''
CREATE TABLE IF NOT EXISTS var_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATETIME,
    time_horizon_start DATE,
    time_horizon_end DATE,
    var_historical_95 FLOAT,
    var_variance_covariance_95 FLOAT,
    var_monte_carlo_95 FLOAT,
    var_historical_99 FLOAT,
    var_variance_covariance_99 FLOAT,
    var_monte_carlo_99 FLOAT
)
''')

mydb.commit()  # Salvataggio dei cambiamenti nel database

# Chiudere la connessione al database
mydb.close()
