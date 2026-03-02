"""
ml/ — Módulo de predicción para Google Trends
──────────────────────────────────────────────
Uso básico:
    from ml.forecaster import Forecaster

    fc = Forecaster()
    fc.train(keyword='creatine')
    fc.summary()
    fc.plot(weeks=26, save_path='figures/forecast/creatine.png')
"""

from ml.forecaster import Forecaster, plot_all_forecasts

__all__ = ["Forecaster", "plot_all_forecasts"]