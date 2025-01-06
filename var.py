import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import mysql.connector
from datetime import datetime




def var_montecarlo(data, confidence_level=0.99, p=True):
    returns = data['Close'].pct_change().dropna()

    # Simula i rendimenti futuri utilizzando il metodo Monte Carlo
    num_simulations = 10000
    simulation_horizon = 252  #numero dei giorni di trading in un anno
    simulated_returns = np.random.normal(np.mean(returns), np.std(returns), (simulation_horizon, num_simulations))

    # Calcola i valori simulati di un portafoglio
    initial_investment = 1000000  # $1,000,000
    portfolio_values = initial_investment * np.exp(np.cumsum(simulated_returns, axis=0))

    # Calcola il rendimento del portafoglio
    portfolio_returns = portfolio_values[-1] / portfolio_values[0] - 1

    VaR_monte_carlo = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    if p:
        # Traccia la distribuzione dei rendimenti simulati del portafoglio e la soglia del VaR
        plt.figure(figsize=(10, 6))
        plt.hist(portfolio_returns, bins=50, alpha=0.75, color='orange', edgecolor='black')
        plt.axvline(VaR_monte_carlo, color='red', linestyle='--', label=f'VaR: {VaR_monte_carlo:.2%}')
        plt.title('Simulated Portfolio Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend()

        plt.savefig(f'var_montecarlo{confidence_level}.png')

        plt.show()
    return VaR_monte_carlo

def var_varianza_covarianza(data,confidence_level=0.99, p=True):
    returns = data['Close'].pct_change().dropna()

    # Calcola la media e la deviazione standard dei ritorni
    mean_return = returns.mean()  
    std_dev = returns.std()       

    z_score = norm.ppf(1 - confidence_level)
    VaR_variance_covariance = mean_return + z_score * std_dev

    # Evita l'uso del float() per la Serie, usa il primo elemento se necessario
    VaR_variance_covariance = VaR_variance_covariance if isinstance(VaR_variance_covariance, float) else VaR_variance_covariance.iloc[0]
    
    if p:
        # Crea il grafico
        plt.figure(figsize=(10, 6))

        # Correggi la forma di x e y per essere unidimensionali
        x = np.linspace(mean_return - 3*std_dev, mean_return + 3*std_dev, 1000)  # Array unidimensionale
        y = norm.pdf(x, mean_return, std_dev)  # Densità di probabilità

        # Assicurati che x e y siano unidimensionali
        x = x.flatten()  # Forza x a essere unidimensionale
        y = y.flatten()  # Forza y a essere unidimensionale



        # Maschera booleana per la parte sotto la soglia VaR
        mask = x <= VaR_variance_covariance

        # Grafico della distribuzione normale e della soglia VaR
        plt.plot(x, y, label='Distribuzione Normale')
        plt.axvline(VaR_variance_covariance, color= 'pink', linestyle='--', label=f'VaR: {VaR_variance_covariance:.2%}')

        # Assicurati che x e y siano unidimensionali
        plt.fill_between(x, 0, y, where=mask, color='pink', alpha=0.5)

        plt.title('Distribuzione Normale dei Rendimenti con Soglia VaR')
        plt.xlabel('Rendimenti')
        plt.ylabel('Densità di Probabilità')
        plt.legend()
        plt.savefig(f'var_varianza-covarianza{confidence_level}.png')
        plt.show()
    return VaR_variance_covariance

def var_metodo_storico(data, confidence_level=0.99, p=True):
    returns = data['Close'].pct_change().dropna()

    VaR_historical = np.percentile(returns, (1 - confidence_level) * 100)
    if p:
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=50, alpha=0.75, color='red', edgecolor='black')
        plt.axvline(VaR_historical, color='red', linestyle='--', label=f'VaR: {VaR_historical:.2%}')
        plt.title('Historical Returns of ISP.MI')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend()

        plt.savefig(f'var_metodo_storico{confidence_level}.png')

        plt.show()
    return VaR_historical



def insertintodb(data):

    # Connessione al database
    mydb = mysql.connector.connect(
        host="localhost",  
        user="root",  
        password="",  
        database="VaR_database"  
    )
    cursor = mydb.cursor() 

    # Preparazione dei dati per l'inserimento
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    horizon_time_start = data.index.min().strftime("%Y-%m-%d")  
    horizon_time_end = data.index.max().strftime("%Y-%m-%d")

    var_historical_95 = var_metodo_storico(data, 0.95,False)
    var_variance_covariance_95 = var_varianza_covarianza(data, 0.95,False)
    var_monte_carlo_95 = var_montecarlo(data, 0.95,False)

    # Calcolare VaR per ogni metodo con 99% di confidenza
    var_historical_99 = var_metodo_storico(data, 0.99, False)
    var_variance_covariance_99 = var_varianza_covarianza(data, 0.99, False)
    var_monte_carlo_99 = var_montecarlo(data, 0.99, False)
    # Inserimento dei risultati nel database
    cursor.execute('''
    INSERT INTO var_results (date, time_horizon_start, time_horizon_end, var_historical_95, var_variance_covariance_95, var_monte_carlo_95, 
                            var_historical_99, var_variance_covariance_99, var_monte_carlo_99)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (date, horizon_time_start, horizon_time_end, float(var_historical_95), float(var_variance_covariance_95), float(var_monte_carlo_95),
          float(var_historical_99), float(var_variance_covariance_99), float(var_monte_carlo_99)))
    
    
    mydb.commit()  

    mydb.close()
    return True



if __name__ == "__main__":

    #recupero dati storici
    data = yf.download("ISP.MI", start="2023-01-01", end="2025-01-01")
   
   
    #calcolo var montecarlo con livello di confidenza al 95%
    var_montecarlo95 = var_montecarlo(data, 0.95)
    print(f"Il VaR calcolato con il metodo Monte Carlo con un intervallo di confidenza di 95% è {var_montecarlo95:.2%}")


    #calcolo var montecarlo con livello di confidenza al 99%
    var_montecarlo99 = var_montecarlo(data, 0.99)
    print(f"Il VaR calcolato con il metodo Monte Carlo con un intervallo di confidenza del 99% è {var_montecarlo99:.2%}")


    #calcolo var varianza-covarianza con livello di confidenza al 95%
    var_varcov95 = var_varianza_covarianza(data, 0.95)
    print(f"Il VaR con il metodo varianza-covarianza con intervallo di confidenza 95% di Intesa Sanpaolo è {var_varcov95:.2%}")


    #calcolo var varianza-covarianza con livello di confidenza al 99%
    var_varcov99 = var_varianza_covarianza(data, 0.99)
    print(f"Il VaR con il metodo varianza-covarianza con intervallo di confidenza 99% di Intesa Sanpaolo è {var_varcov99:.2%}")


    #calcolo var storico con livello di confidenza al 95%
    var_storico95 = var_metodo_storico(data, 0.95)
    print(f"Il VaR con il metodo storico con intervallo di confidenza al 95% di Intesa Sanpaolo è {var_storico95:.2%}")
   
    #calcolo var storico con livello di confidenza al 99%
    var_storico99 = var_metodo_storico(data, 0.99)
    print(f"Il VaR con il metodo storico con intervallo di confidenza al 99% di Intesa Sanpaolo è {var_storico99:.2%}")

    #inserisci nel db
    insertintodb(data)