import pandas as pd
import numpy as np
import pandas_datareader.data as web
from arch import arch_model

def fit_team_aparch_t(returns_series: pd.Series, forecast_horizon: int = 1) -> dict:
    """
    Fits an APARCH(1,1) model with Student's t-errors to a % scaled return series.
    """
    # Fit APARCH(1,1) with Student's t-distribution
    model = arch_model(
        returns_series,
        mean='Constant',
        vol='APARCH',
        p=1, o=1, q=1,
        dist='StudentsT'  # Captures heavy tails via estimated degrees of freedom (nu)
    )
    
    results = model.fit(disp='off')
    
    # Structure parameters alongside p-values for diagnostic checks
    fitted_parameters = {
        param: {
            "val": float(results.params[param]),
            "pvalue": float(results.pvalues[param])
        }
        for param in results.params.index
    }
    
    # Out-of-sample volatility forecast
    forecast_obj = results.forecast(horizon=forecast_horizon)
    raw_variance_forecast = forecast_obj.variance.iloc[-1].values
    volatility_forecast = np.sqrt(raw_variance_forecast)
    
    last_date = returns_series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='B')[1:]
    
    labeled_volatility_forecast = {
        date.strftime('%Y-%m-%d'): float(vol) 
        for date, vol in zip(future_dates, volatility_forecast)
    }
    
    return {
        "parameters": fitted_parameters,
        "log_likelihood": results.loglikelihood,
        "aic": results.aic,
        "bic": results.bic,
        "forecasted_volatility": labeled_volatility_forecast
    }

if __name__ == "__main__":
    # 1. Fetch & clean WTI oil data
    start, end = "2006-01-01", "2025-12-31"
    wti = web.DataReader("DCOILWTICO", "fred", start, end)
    wti.columns = ["price"]
    wti.index = pd.to_datetime(wti.index)
    
    wti = wti[wti['price'] > 0].dropna()
    returns = 100 * np.log(wti['price'] / wti['price'].shift(1)).dropna()
    returns.name = "log_returns_pct"

    # 2. Fit Student's t APARCH
    output = fit_team_aparch_t(returns, forecast_horizon=1)

    print("--- Model Performance (Student's t APARCH) ---")
    print(f"Log-Likelihood: {output['log_likelihood']:.4f}")
    print(f"AIC:            {output['aic']:.4f}")
    print(f"BIC:            {output['bic']:.4f}\n")

    print("--- Fitted Parameters & Significance ---")
    print(f"{'Parameter':<10} | {'Value':<10} | {'p-value':<10} | {'Status'}")
    print("-" * 48)
    for param, metrics in output['parameters'].items():
        sig = "Significant" if metrics['pvalue'] < 0.05 else "Not Significant"
        print(f"{param:<10} | {metrics['val']:<10.6f} | {metrics['pvalue']:<10.4e} | {sig}")
        
    print("\n--- Dated 1-Day Volatility Forecast ---")
    for date, vol in output['forecasted_volatility'].items():
        print(f"{date}: {vol:.6f}% daily std dev")

