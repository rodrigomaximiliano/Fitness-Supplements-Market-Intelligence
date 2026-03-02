"""
ml/evaluator.py
───────────────
Evalúa qué tan bien predice el modelo comparando sus predicciones
contra datos reales que él nunca vio durante el entrenamiento.

Cómo funciona (sin tecnicismos):
    Tomamos los últimos N semanas del histórico, las "escondemos",
    entrenamos el modelo sin ellas, y le preguntamos que las prediga.
    Después comparamos lo que predijo contra lo que realmente pasó.
    Si el error es bajo, el modelo es confiable.

Ejemplo de uso:
    from ml.evaluator import Evaluator

    ev = Evaluator()
    ev.evaluate(keyword='creatine', test_weeks=12)
    ev.report()
    ev.plot(save_path='figures/forecast/creatine_eval.png')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from ml.forecaster import Forecaster, COLORS


# ── Clase principal ───────────────────────────────────────────────────────────
class Evaluator:
    """
    Evalúa la calidad del modelo usando una técnica llamada
    'hold-out validation':

        1. Divide los datos en dos partes:
             - ENTRENAMIENTO: todo excepto las últimas N semanas
             - TEST:          las últimas N semanas (datos "escondidos")

        2. Entrena el modelo SOLO con los datos de entrenamiento

        3. Le pide al modelo que prediga las semanas del test

        4. Compara predicción vs realidad con tres métricas:
             - MAE:      error promedio en puntos de índice
             - RMSE:     similar al MAE pero penaliza más los errores grandes
             - MAPE:     error expresado en porcentaje (más fácil de explicar)

    Atributos públicos después de evaluate():
        keyword       - keyword evaluada
        test_weeks    - cuántas semanas se usaron como test
        df_train      - datos usados para entrenar
        df_test       - datos reales del período de test
        df_pred       - predicciones del modelo para ese período
        mae           - error promedio (puntos de índice)
        rmse          - error cuadrático medio
        mape          - error porcentual medio
        accuracy      - 100 - mape (qué tan "acertado" fue el modelo)
    """

    def __init__(self):
        self.keyword    = None
        self.test_weeks = None
        self.df_train   = None
        self.df_test    = None
        self.df_pred    = None
        self.mae        = None
        self.rmse       = None
        self.mape       = None
        self.accuracy   = None

    # ── Evaluación ────────────────────────────────────────────────────────────
    def evaluate(self, keyword: str = "creatine", test_weeks: int = 12) -> "Evaluator":
        """
        Ejecuta la evaluación completa.

        Parámetros:
            keyword    - keyword a evaluar
            test_weeks - cuántas semanas reservar para test (default 12 = 3 meses)

        Devuelve self para encadenar: ev.evaluate().report().plot()
        """
        self.keyword    = keyword
        self.test_weeks = test_weeks

        # ── Cargar datos completos ────────────────────────────────────────────
        fc_full = Forecaster()
        fc_full.train(keyword=keyword)
        df_full = fc_full.df_history.copy()

        # ── Dividir: entrenamiento / test ─────────────────────────────────────
        self.df_train = df_full.iloc[:-test_weeks].copy().reset_index(drop=True)
        self.df_test  = df_full.iloc[-test_weeks:].copy().reset_index(drop=True)

        # ── Entrenar modelo SOLO con datos de entrenamiento ───────────────────
        # Guardamos el CSV original y lo reemplazamos temporalmente
        # Para evitar tocar el archivo, entrenamos manualmente
        fc_eval = _ForecasterOnDataframe()
        fc_eval.train_on_df(self.df_train, keyword=keyword)

        # ── Predecir las semanas del período de test ──────────────────────────
        X_test = fc_eval._make_future_features(test_weeks)
        y_pred = fc_eval.model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 100)

        # Construir DataFrame de predicciones con las fechas reales del test
        uncertainty = fc_eval._residual_std * np.sqrt(np.arange(1, test_weeks + 1))
        self.df_pred = pd.DataFrame({
            "week":       self.df_test["week"].values,
            "prediction": y_pred,
            "lower":      np.clip(y_pred - uncertainty, 0, 100),
            "upper":      np.clip(y_pred + uncertainty, 0, 100),
        })

        # ── Calcular métricas ─────────────────────────────────────────────────
        y_real = self.df_test[keyword].values.astype(float)
        y_hat  = self.df_pred["prediction"].values

        self.mae  = mean_absolute_error(y_real, y_hat)
        self.rmse = root_mean_squared_error(y_real, y_hat)

        # MAPE: evitar división por cero para keywords con valores muy bajos
        mask = y_real > 1
        if mask.sum() > 0:
            self.mape = np.mean(np.abs((y_real[mask] - y_hat[mask]) / y_real[mask])) * 100
        else:
            self.mape = 0.0

        self.accuracy = max(0, 100 - self.mape)

        return self

    # ── Reporte ───────────────────────────────────────────────────────────────
    def report(self) -> "Evaluator":
        """
        Imprime un resumen legible de los resultados de evaluación.
        """
        if self.mae is None:
            raise RuntimeError("Llamá a evaluate() primero.")

        stars = "*" * self._star_rating()

        print(f"\n{'─'*52}")
        print(f"  EVALUACIÓN DEL MODELO — {self.keyword.upper()}")
        print(f"{'─'*52}")
        print(f"  Período de test:   últimas {self.test_weeks} semanas")
        print(f"  Datos entrenamiento: {len(self.df_train)} semanas")
        print(f"")
        print(f"  MAE:       {self.mae:.2f}  pts de índice en promedio")
        print(f"  RMSE:      {self.rmse:.2f}  pts (penaliza errores grandes)")
        print(f"  MAPE:      {self.mape:.1f}%  error porcentual medio")
        print(f"  Precisión: {self.accuracy:.1f}%  {stars}")
        print(f"")
        print(f"  Interpretación: {self._interpretation()}")
        print(f"{'─'*52}\n")

        return self

    def _star_rating(self) -> int:
        """Convierte MAPE en estrellas (1-5)."""
        if self.mape < 5:   return 5
        if self.mape < 10:  return 4
        if self.mape < 20:  return 3
        if self.mape < 35:  return 2
        return 1

    def _interpretation(self) -> str:
        """Texto explicativo según el MAPE."""
        if self.mape < 5:
            return "Excelente — el modelo predice con muy alta precisión."
        if self.mape < 10:
            return "Bueno — errores pequeños, confiable para decisiones."
        if self.mape < 20:
            return "Aceptable — sirve para ver la dirección, no el número exacto."
        if self.mape < 35:
            return "Limitado — usarlo solo para tendencia general."
        return "Bajo — el patrón histórico no es suficiente para predecir."

    # ── Gráfico ───────────────────────────────────────────────────────────────
    def plot(self, save_path: str = None, show: bool = True) -> None:
        """
        Genera el gráfico de evaluación:
          - Línea gris: histórico de entrenamiento
          - Línea color: predicción del modelo
          - Línea blanca punteada: valores reales del período de test
          - Área sombreada: intervalo de confianza

        Ver si la línea blanca (real) cae dentro del área sombreada (predicción)
        es la forma visual de saber si el modelo es bueno.
        """
        if self.mae is None:
            raise RuntimeError("Llamá a evaluate() primero.")

        color = COLORS.get(self.keyword, "#E94560")

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#1A1A2E")
        ax.set_facecolor("#1A1A2E")

        # ── Histórico de entrenamiento ────────────────────────────────────────
        ax.plot(
            self.df_train["week"],
            self.df_train[self.keyword],
            color="#94A3B8",
            linewidth=1.5,
            label="Entrenamiento (histórico)",
            zorder=2
        )

        # ── Línea de corte ────────────────────────────────────────────────────
        ax.axvline(
            x=self.df_train["week"].iloc[-1],
            color="white",
            linewidth=0.8,
            linestyle="--",
            alpha=0.4,
            zorder=3
        )

        # ── Intervalo de confianza ────────────────────────────────────────────
        ax.fill_between(
            self.df_pred["week"],
            self.df_pred["lower"],
            self.df_pred["upper"],
            color=color,
            alpha=0.2,
            label="Intervalo de confianza",
            zorder=1
        )

        # ── Predicción ────────────────────────────────────────────────────────
        ax.plot(
            self.df_pred["week"],
            self.df_pred["prediction"],
            color=color,
            linewidth=2.5,
            label="Predicción del modelo",
            zorder=4
        )

        # ── Valores REALES del período de test ────────────────────────────────
        ax.plot(
            self.df_test["week"],
            self.df_test[self.keyword],
            color="white",
            linewidth=2,
            linestyle=":",
            label="Valores reales (test)",
            zorder=5
        )

        # ── Título y estilo ───────────────────────────────────────────────────
        ax.set_title(
            f"{self.keyword.replace('_', ' ').upper()}  —  Evaluación del modelo  "
            f"(test: {self.test_weeks} semanas)",
            color="white", fontsize=13, fontweight="bold", pad=16
        )
        ax.set_ylabel("Índice Google Trends (0–100)", color="#94A3B8", fontsize=11)
        ax.set_ylim(0, 110)

        ax.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.6)
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30, ha="right", color="#94A3B8")

        ax.legend(
            facecolor="#0F3460", edgecolor="#334155",
            labelcolor="white", fontsize=10
        )

        # ── Métricas en el gráfico ────────────────────────────────────────────
        stars = "*" * self._star_rating()
        ax.text(
            0.01, 0.96,
            f"MAE: ±{self.mae:.1f} pts  |  Precisión: {self.accuracy:.1f}%  {stars}",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            va="top",
            bbox=dict(facecolor="#0F3460", edgecolor="#334155",
                      boxstyle="round,pad=0.4", alpha=0.8)
        )

        # ── Leyenda de cómo leer el gráfico ──────────────────────────────────
        ax.text(
            0.01, 0.06,
            "Si la línea blanca cae dentro del área sombreada → modelo confiable",
            transform=ax.transAxes,
            color="#64748B",
            fontsize=9,
            style="italic"
        )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor()
            )
            print(f"💾  Guardado en {save_path}")

        if show:
            plt.show()

        plt.close()

    # ── Comparar todas las keywords ───────────────────────────────────────────
    @staticmethod
    def compare_all(test_weeks: int = 12) -> pd.DataFrame:
        """
        Evalúa todas las keywords y devuelve una tabla comparativa.

        Útil para ver de un vistazo cuáles predicciones son más confiables.
        """
        from ml.forecaster import KEYWORDS

        results = []
        for kw in KEYWORDS.keys():
            try:
                ev = Evaluator()
                ev.evaluate(keyword=kw, test_weeks=test_weeks)
                results.append({
                    "keyword":   kw,
                    "MAE":       round(ev.mae, 2),
                    "RMSE":      round(ev.rmse, 2),
                    "MAPE (%)":  round(ev.mape, 1),
                    "Precisión": f"{ev.accuracy:.1f}%",
                    "Rating":    "*" * ev._star_rating(),
                })
            except Exception as e:
                results.append({"keyword": kw, "error": str(e)})

        df = pd.DataFrame(results)
        print("\n📊  COMPARACIÓN DE MODELOS\n")
        print(df.to_string(index=False))
        print()
        return df


# ── Helper interno ─────────────────────────────────────────────────────────────
class _ForecasterOnDataframe(Forecaster):
    """
    Versión interna del Forecaster que acepta un DataFrame directamente
    en lugar de cargar el CSV. Permite dividir los datos para evaluación
    sin tocar el archivo original.
    """

    def train_on_df(self, df: pd.DataFrame, keyword: str) -> None:
        """
        Igual que Forecaster.train() pero recibe el DataFrame ya preparado.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error

        self.keyword    = keyword
        self.df_history = df.copy()

        y = df[keyword].values.astype(float)
        X = self._make_features(df)

        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg",  LinearRegression())
        ])
        self.model.fit(X, y)

        y_pred = self.model.predict(X)
        self.mae  = mean_absolute_error(y, y_pred)
        self.rmse = root_mean_squared_error(y, y_pred)

        residuals = y - y_pred
        self._residual_std = residuals.std()