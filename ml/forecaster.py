"""
ml/forecaster.py
────────────────
Predice el índice de Google Trends para las próximas N semanas.

Cómo funciona (sin tecnicismos):
    1. Carga los datos históricos semanales
    2. Le enseña al modelo DOS cosas:
         - La tendencia general: creatina viene subiendo desde 2021
         - La estacionalidad: enero siempre pica, diciembre siempre baja
    3. Combina ambas para predecir semanas futuras
    4. Devuelve la predicción con un rango de confianza (mínimo y máximo)

Ejemplo de uso:
    from ml.forecaster import Forecaster

    fc = Forecaster()
    fc.train(keyword='creatine')
    df_pred = fc.predict(weeks=26)   # próximas 26 semanas
    fc.plot(keyword='creatine', save_path='figures/forecast/creatine.png')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# ── Constantes ────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "multiTimeline.csv"

KEYWORDS = {
    "creatine":       "creatine: (Estados Unidos)",
    "protein_powder": "protein powder: (Estados Unidos)",
    "pre_workout":    "pre workout: (Estados Unidos)",
    "omega_3":        "omega 3: (Estados Unidos)",
    "weight_loss":    "weight loss supplements: (Estados Unidos)",
}

COLORS = {
    "creatine":       "#E94560",
    "protein_powder": "#4A90D9",
    "pre_workout":    "#F59E0B",
    "omega_3":        "#10B981",
    "weight_loss":    "#94A3B8",
}


# ── Clase principal ───────────────────────────────────────────────────────────
class Forecaster:
    """
    Entrena un modelo de predicción para una keyword de Google Trends
    y genera predicciones con intervalo de confianza.

    Atributos públicos después de train():
        keyword      - nombre de la keyword
        df_history   - DataFrame con los datos históricos
        model        - pipeline entrenado (tendencia + estacionalidad)
        mae          - error promedio del modelo en datos históricos
        rmse         - error cuadrático medio del modelo
    """

    def __init__(self):
        self.keyword     = None
        self.df_history  = None
        self.model       = None
        self.mae         = None
        self.rmse        = None
        self._residual_std = None   # para calcular el intervalo de confianza

    # ── Carga de datos ────────────────────────────────────────────────────────
    def _load_data(self) -> pd.DataFrame:
        """
        Lee el CSV de Google Trends (tiene 2 filas de header que hay que saltar).
        Devuelve un DataFrame limpio con columna 'week' como datetime.
        """
        df = pd.read_csv(DATA_PATH, skiprows=2)

        # Renombrar columnas a nombres cortos
        col_map = {"Semana": "week"}
        for short, full in KEYWORDS.items():
            # El CSV puede tener el nombre exacto o con espacios diferentes
            for col in df.columns:
                if short.replace("_", " ") in col.lower() or full.lower() in col.lower():
                    col_map[col] = short
        df = df.rename(columns=col_map)
        df["week"] = pd.to_datetime(df["week"])
        df = df.sort_values("week").reset_index(drop=True)
        return df

    # ── Feature engineering ───────────────────────────────────────────────────
    def _make_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convierte las fechas en números que el modelo puede entender.

        Columnas que genera:
          - t            : número de semana (0, 1, 2, ... 261)
          - sin_52, cos_52 : patrón anual (sube en enero, baja en diciembre)
          - sin_26, cos_26 : patrón semestral (el pico de julio)

        Por qué seno y coseno: son funciones que se repiten cíclicamente,
        igual que las estaciones del año. El modelo aprende qué tan fuerte
        es cada ciclo y en qué punto del año ocurre.
        """
        t = np.arange(len(df)).reshape(-1, 1)

        # Ciclo de 52 semanas = 1 año
        sin_52 = np.sin(2 * np.pi * t / 52)
        cos_52 = np.cos(2 * np.pi * t / 52)

        # Ciclo de 26 semanas = 6 meses (captura el pico de julio)
        sin_26 = np.sin(2 * np.pi * t / 26)
        cos_26 = np.cos(2 * np.pi * t / 26)

        return np.hstack([t, sin_52, cos_52, sin_26, cos_26])

    def _make_future_features(self, n_weeks: int) -> np.ndarray:
        """
        Genera las mismas features pero para semanas futuras.
        Arranca desde len(df) para continuar donde terminaron los datos.
        """
        n_hist = len(self.df_history)
        t = np.arange(n_hist, n_hist + n_weeks).reshape(-1, 1)

        sin_52 = np.sin(2 * np.pi * t / 52)
        cos_52 = np.cos(2 * np.pi * t / 52)
        sin_26 = np.sin(2 * np.pi * t / 26)
        cos_26 = np.cos(2 * np.pi * t / 26)

        return np.hstack([t, sin_52, cos_52, sin_26, cos_26])

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    def train(self, keyword: str = "creatine") -> "Forecaster":
        """
        Entrena el modelo con todos los datos históricos disponibles.

        Parámetro:
            keyword - nombre corto: 'creatine', 'protein_powder', etc.

        Devuelve self para poder encadenar: fc.train().predict()
        """
        if keyword not in KEYWORDS:
            raise ValueError(f"Keyword '{keyword}' no válida. Opciones: {list(KEYWORDS.keys())}")

        self.keyword    = keyword
        self.df_history = self._load_data()

        if keyword not in self.df_history.columns:
            raise ValueError(f"Columna '{keyword}' no encontrada en el CSV.")

        y = self.df_history[keyword].values.astype(float)
        X = self._make_features(self.df_history)

        # Pipeline: features polinomiales (captura curvas) + regresión lineal
        # degree=2 es suficiente — más alto y el modelo se "sobreajusta"
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg",  LinearRegression())
        ])
        self.model.fit(X, y)

        # Evaluar qué tan bien ajusta sobre datos históricos
        y_pred = self.model.predict(X)
        self.mae  = mean_absolute_error(y, y_pred)
        self.rmse = root_mean_squared_error(y, y_pred)

        # Desviación estándar de los residuos = base del intervalo de confianza
        residuals = y - y_pred
        self._residual_std = residuals.std()

        print(f"✅  Modelo entrenado para '{keyword}'")
        print(f"    MAE:  {self.mae:.2f} puntos de índice")
        print(f"    RMSE: {self.rmse:.2f} puntos de índice")
        print(f"    Intervalo de confianza (±1σ): ±{self._residual_std:.2f}")

        return self

    # ── Predicción ────────────────────────────────────────────────────────────
    def predict(self, weeks: int = 26) -> pd.DataFrame:
        """
        Predice las próximas N semanas.

        Devuelve un DataFrame con:
            week        - fecha de cada semana futura
            prediction  - valor predicho (índice 0-100)
            lower       - límite inferior del intervalo de confianza
            upper       - límite superior del intervalo de confianza

        El intervalo usa ±1 desviación estándar de los errores históricos.
        Cuanto más lejos en el futuro, más ancho el rango (incertidumbre crece).
        """
        if self.model is None:
            raise RuntimeError("Llamá a train() primero.")

        X_future = self._make_future_features(weeks)
        y_pred   = self.model.predict(X_future)

        # La incertidumbre crece con el tiempo:
        # para la semana 1 es ±1σ, para la semana 26 es ±√26 σ
        uncertainty = self._residual_std * np.sqrt(np.arange(1, weeks + 1))

        # Generar fechas futuras (continúan desde la última semana histórica)
        last_date    = self.df_history["week"].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=weeks,
            freq="W-SUN"
        )

        df_pred = pd.DataFrame({
            "week":       future_dates,
            "prediction": np.clip(y_pred, 0, 100),          # índice siempre 0-100
            "lower":      np.clip(y_pred - uncertainty, 0, 100),
            "upper":      np.clip(y_pred + uncertainty, 0, 100),
        })

        return df_pred

    # ── Resumen ───────────────────────────────────────────────────────────────
    def summary(self, weeks: int = 26) -> None:
        """
        Imprime un resumen legible de la predicción.
        """
        if self.model is None:
            raise RuntimeError("Llamá a train() primero.")

        df_pred = self.predict(weeks)
        last_hist = self.df_history[self.keyword].iloc[-1]

        print(f"\n📈  PREDICCIÓN — {self.keyword.upper()}")
        print(f"    Último valor histórico:  {last_hist:.0f}  ({self.df_history['week'].iloc[-1].date()})")
        print(f"    En 4 semanas:   {df_pred['prediction'].iloc[3]:.0f}  (rango {df_pred['lower'].iloc[3]:.0f}–{df_pred['upper'].iloc[3]:.0f})")
        print(f"    En 13 semanas:  {df_pred['prediction'].iloc[12]:.0f}  (rango {df_pred['lower'].iloc[12]:.0f}–{df_pred['upper'].iloc[12]:.0f})")
        print(f"    En 26 semanas:  {df_pred['prediction'].iloc[25]:.0f}  (rango {df_pred['lower'].iloc[25]:.0f}–{df_pred['upper'].iloc[25]:.0f})")
        print(f"\n    Tendencia: {'📈 SUBE' if df_pred['prediction'].iloc[-1] > last_hist else '📉 BAJA'}")

    # ── Gráfico ───────────────────────────────────────────────────────────────
    def plot(
        self,
        weeks: int = 26,
        keyword: str = None,
        save_path: str = None,
        show: bool = True
    ) -> None:
        """
        Genera el gráfico con:
          - Línea gris:  datos históricos (5 años)
          - Línea color: predicción
          - Área sombreada: intervalo de confianza

        Parámetros:
            weeks     - semanas a predecir (default 26 = 6 meses)
            save_path - ruta donde guardar el PNG (opcional)
            show      - mostrar en pantalla (default True)
        """
        if self.model is None:
            raise RuntimeError("Llamá a train() primero.")

        kw      = keyword or self.keyword
        df_pred = self.predict(weeks)
        color   = COLORS.get(kw, "#E94560")

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#1A1A2E")
        ax.set_facecolor("#1A1A2E")

        # ── Histórico ─────────────────────────────────────────────────────────
        ax.plot(
            self.df_history["week"],
            self.df_history[kw],
            color="#94A3B8",
            linewidth=1.5,
            label="Histórico (2021–2026)",
            zorder=2
        )

        # ── Línea vertical de separación ──────────────────────────────────────
        ax.axvline(
            x=self.df_history["week"].iloc[-1],
            color="white",
            linewidth=0.8,
            linestyle="--",
            alpha=0.4,
            zorder=3
        )

        # ── Predicción ────────────────────────────────────────────────────────
        ax.plot(
            df_pred["week"],
            df_pred["prediction"],
            color=color,
            linewidth=2.5,
            label=f"Predicción ({weeks} semanas)",
            zorder=4
        )

        # ── Intervalo de confianza ────────────────────────────────────────────
        ax.fill_between(
            df_pred["week"],
            df_pred["lower"],
            df_pred["upper"],
            color=color,
            alpha=0.15,
            label="Intervalo de confianza",
            zorder=1
        )

        # ── Anotación del último valor predicho ───────────────────────────────
        last_pred = df_pred["prediction"].iloc[-1]
        last_date = df_pred["week"].iloc[-1]
        ax.annotate(
            f"  {last_pred:.0f}",
            xy=(last_date, last_pred),
            color=color,
            fontsize=11,
            fontweight="bold",
            va="center"
        )

        # ── Estilo ────────────────────────────────────────────────────────────
        ax.set_title(
            f"{kw.replace('_', ' ').upper()}  —  Predicción próximas {weeks} semanas",
            color="white", fontsize=14, fontweight="bold", pad=16
        )
        ax.set_ylabel("Índice Google Trends (0–100)", color="#94A3B8", fontsize=11)
        ax.set_ylim(0, 110)
        ax.set_xlim(
            self.df_history["week"].iloc[0],
            df_pred["week"].iloc[-1] + pd.Timedelta(weeks=2)
        )

        # Grilla y ejes
        ax.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.6)
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30, ha="right", color="#94A3B8")

        legend = ax.legend(
            facecolor="#0F3460",
            edgecolor="#334155",
            labelcolor="white",
            fontsize=10
        )

        # MAE en esquina
        ax.text(
            0.01, 0.04,
            f"MAE histórico: ±{self.mae:.1f} pts",
            transform=ax.transAxes,
            color="#64748B",
            fontsize=9
        )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"💾  Guardado en {save_path}")

        if show:
            plt.show()

        plt.close()


# ── Comparar múltiples keywords ───────────────────────────────────────────────
def plot_all_forecasts(weeks: int = 26, save_dir: str = "figures/forecast") -> None:
    """
    Entrena y grafica predicciones para todas las keywords en un solo plot.
    Útil para la presentación — muestra todas las tendencias juntas.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    for kw, color in COLORS.items():
        try:
            fc = Forecaster()
            fc.train(keyword=kw)
            df_hist = fc.df_history
            df_pred = fc.predict(weeks)

            # Histórico en tono suave
            ax.plot(df_hist["week"], df_hist[kw],
                    color=color, linewidth=1.2, alpha=0.5)

            # Predicción más gruesa
            ax.plot(df_pred["week"], df_pred["prediction"],
                    color=color, linewidth=2, linestyle="--",
                    label=kw.replace("_", " ").title())

        except Exception as e:
            print(f"⚠️  Error en {kw}: {e}")

    # Línea de separación histórico/predicción
    last_hist_date = pd.read_csv(DATA_PATH, skiprows=2)
    ax.axvline(x=pd.to_datetime(
        pd.read_csv(DATA_PATH, skiprows=2).iloc[-1, 0]),
        color="white", linewidth=0.8, linestyle="--", alpha=0.4
    )

    ax.set_title("TODAS LAS KEYWORDS — Histórico + Predicción",
                 color="white", fontsize=14, fontweight="bold", pad=16)
    ax.set_ylabel("Índice Google Trends (0–100)", color="#94A3B8")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.6)
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right", color="#94A3B8")

    ax.legend(facecolor="#0F3460", edgecolor="#334155",
              labelcolor="white", fontsize=10, loc="upper left")

    plt.tight_layout()
    out = f"{save_dir}/all_keywords_forecast.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"💾  Guardado en {out}")
    plt.close()