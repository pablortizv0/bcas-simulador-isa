from typing import List, Literal, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from numpy_financial import irr
import unicodedata


# ============================================
# Tablas base y mapeos
# ============================================

# Empleabilidad acumulada % (mes 1..36)
EMPLEABILIDAD_TABLA_DEFAULT = {
    'A+': [8.1, 12.1, 17.8, 25.2, 34.5, 45.0, 55.8, 66.0, 74.6, 81.4, 86.5, 90.0, 92.4, 94.0, 95.1, 95.8, 96.2, 96.5, 96.7, 96.8, 96.9, 96.9, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0],
    'A' : [5.8, 8.5, 12.1, 17.0, 23.3, 31.0, 39.8, 49.3, 58.6, 67.0, 74.2, 80.0, 84.4, 87.6, 89.9, 91.5, 92.7, 93.4, 93.9, 94.3, 94.5, 94.7, 94.8, 94.9, 94.9, 94.9, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0],
    'B' : [6.0, 8.3, 11.4, 15.3, 20.3, 26.4, 33.4, 41.1, 49.0, 56.7, 63.8, 70.0, 75.1, 79.2, 82.4, 84.7, 86.5, 87.8, 88.7, 89.4, 89.8, 90.2, 90.4, 90.6, 90.7, 90.8, 90.9, 90.9, 90.9, 91.0, 91.0, 91.0, 91.0, 91.0, 91.0, 91.0],
    'C' : [5.4, 7.2, 9.7, 12.8, 16.6, 21.4, 26.9, 33.2, 40.0, 46.9, 53.7, 60.0, 65.6, 70.3, 74.2, 77.3, 79.8, 81.6, 83.0, 84.1, 84.9, 85.4, 85.9, 86.2, 86.4, 86.6, 86.7, 86.8, 86.8, 86.9, 86.9, 86.9, 87.0, 87.0, 87.0, 87.0],
    'D' : [5.4, 7.0, 9.0, 11.5, 14.6, 18.3, 22.6, 27.5, 32.8, 38.5, 44.3, 50.0, 55.4, 60.3, 64.6, 68.3, 71.4, 73.9, 75.9, 77.6, 78.8, 79.8, 80.6, 81.1, 81.6, 81.9, 82.2, 82.4, 82.5, 82.7, 82.7, 82.8, 82.9, 82.9, 82.9, 82.9],
    'Prueba': [0]*6 + [95]*30
}

# Mapa (legible → clave)
CATEGORIA_CENTRO_MAP = {
    "Excelente": "A+",
    "Muy Alto": "A",
    "Alto": "B",
    "Medio": "C",
    "Bajo": "D",
    "Prueba": "Prueba",
}

# Salarios anuales por categoría
SALARIO_CATEGORIAS_DEFAULT = {
    'Muy Alto': 45000,
    'Alto': 35000,
    'Medio': 25000,
    'Bajo': 20000,
    'Mínimo': 17000,
    'Otro': 24000,
}


# ============================================
# Modelos Pydantic (I/O)
# ============================================

class TramoRango(BaseModel):
    min_inclusive: float = Field(..., description="Límite inferior del tramo (incluido)")
    max_inclusive: float = Field(..., description="Límite superior del tramo (incluido)")
    descuento: float = Field(..., ge=0.0, le=1.0, description="Descuento decimal (0..1)")

class Payload(BaseModel):
    # ---- Inputs visibles desde V0 ----
    importe_financiado: float = Field(..., gt=0, description="Importe financiado (€)")
    duracion_curso: int = Field(..., ge=0, le=60, description="Duración en meses")
    categoria_centro_legible: Literal["Excelente", "Muy Alto", "Alto", "Medio", "Bajo", "Prueba"] = "Excelente"

    # Tramos de descuento por rangos (min–max)
    descuento_tramos: List[TramoRango] = Field(
        default=[
            TramoRango(min_inclusive=0,       max_inclusive=5000,   descuento=0.05),
            TramoRango(min_inclusive=5000.1,  max_inclusive=10000,  descuento=0.06),
            TramoRango(min_inclusive=10000.1, max_inclusive=15000,  descuento=0.07),
            TramoRango(min_inclusive=15000.1, max_inclusive=30000,  descuento=0.10),
        ],
        description="Rangos de descuento aplicables según Importe Financiado"
    )

    # ---- Parámetros avanzados (configurables desde V0 pero no públicos en UI) ----
    origination_fee: float = 0.10
    take_rate: float = 0.132
    subida_salarial_anual: float = 0.07
    default_rate: float = 0.10
    tir_objetivo: float = 0.05

    pago_concesion: float = 0.10
    # [(porcentaje, mes), ...] — porcentajes sobre el pago total a escuela por estudiante
    pagos_escuela_fijos: List[Tuple[float, int]] = Field(default=[(0.70, 1)])

    categoria_salario: Literal['Muy Alto', 'Alto', 'Medio', 'Bajo', 'Mínimo', 'Otro'] = 'Medio'
    salario_categorias: Optional[dict] = None  # para override opcional

class ISAResult(BaseModel):
    cap_final: float
    ventana1: float
    ventana2: float
    ventana3: float

class Proposal(BaseModel):
    isa: ISAResult


# ============================================
# Helpers de dominio
# ============================================

def _norm(s: str) -> str:
    s2 = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s2.strip().title()

def generar_curva_empleabilidad(categoria_key: str, meses: int, default_rate: float, tabla: dict) -> np.ndarray:
    curva_pct = np.array(tabla[categoria_key][:meses], dtype=float)
    curva_dec = curva_pct / 100.0
    return curva_dec * (1 - default_rate)

def asignar_mes_empleo(curva_empleabilidad: np.ndarray, num_estudiantes: int) -> np.ndarray:
    meses = []
    for i in range(num_estudiantes):
        percentil = (i + 1) / num_estudiantes
        mes = None
        for idx, emp_acum in enumerate(curva_empleabilidad):
            if percentil <= emp_acum:
                mes = idx + 1
                break
        if mes is None and percentil <= curva_empleabilidad[-1]:
            mes = len(curva_empleabilidad)
        meses.append(mes)
    return np.array(meses, dtype=object)

def obtener_descuento_por_rangos(importe: float, tramos: List[TramoRango]) -> float:
    # Busca el primer tramo que contenga el importe (min ≤ importe ≤ max)
    for t in tramos:
        if t.min_inclusive <= importe <= t.max_inclusive:
            return t.descuento
    # Si no entra en ningún tramo, aplica el del último tramo si el importe es mayor
    # (comportamiento tolerante; si prefieres lanzar error, cambia esto por HTTPException)
    if importe > max(t.max_inclusive for t in tramos):
        return tramos[-1].descuento
    raise HTTPException(status_code=422, detail="importe_financiado no entra en ningún rango de descuento_tramos")

def calcular_pago_escuela_total(importe_financiado: float, tramos: List[TramoRango]) -> float:
    desc = obtener_descuento_por_rangos(importe_financiado, tramos)
    return importe_financiado * (1 - desc)


# ============================================
# Simulación core
# ============================================

def simular_estudiante(
    mes_empleo: Optional[int],
    cap_individual: float,
    duracion_curso: int,
    salario_inicial_anual: float,
    subida_salarial_anual: float,
    take_rate: float,
    pago_total_escuela: float,
    pagos_escuela_fijos_pct: List[Tuple[float, int]],  # porcentajes (incluye concesión en mes 1)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
      - cf_escuela_variable: outflow variable a escuela por el residual (% restante) en el mes de empleo
      - cf_inflow_estudiante: inflows de repago (take_rate * salario, limitado por CAP)
    """
    max_meses = 120
    mes_emp_abs = duracion_curso + mes_empleo if mes_empleo is not None else None

    cf_escuela_variable = np.zeros(max_meses)
    cf_inflow_estudiante = np.zeros(max_meses)

    # Pago contingente a la escuela cuando se emplea = residual (1 - suma_fijos)
    if mes_emp_abs is not None and 1 <= mes_emp_abs <= max_meses:
        suma_fijos = sum(p for p, _ in pagos_escuela_fijos_pct)
        residual = max(0.0, 1 - suma_fijos)
        cf_escuela_variable[mes_emp_abs - 1] = residual * pago_total_escuela

    # Inflows del estudiante (repagos)
    if mes_emp_abs is not None:
        salario_mensual_base = salario_inicial_anual / 12.0
        pagado = 0.0
        for m in range(mes_emp_abs, max_meses + 1):
            if pagado >= cap_individual:
                break
            meses_empleado = m - mes_emp_abs
            anios_empleado = meses_empleado // 12
            salario_actual = salario_mensual_base * ((1 + subida_salarial_anual) ** anios_empleado)
            pago = salario_actual * take_rate
            if pagado + pago > cap_individual:
                pago = cap_individual - pagado
            cf_inflow_estudiante[m - 1] = pago
            pagado += pago

    return cf_escuela_variable, cf_inflow_estudiante


def simular_spv(
    cap_individual: float,
    categoria_key: str,
    num_estudiantes: int,
    duracion_curso: int,
    default_rate: float,
    origination_fee: float,
    importe_financiado: float,
    pagos_escuela_fijos_euros_por_est: List[Tuple[float, int]],  # en € por estudiante (mes 1 incluye concesión)
    pagos_escuela_fijos_pct: List[Tuple[float, int]],            # en % por estudiante (mes 1 incluye concesión)
    subida_salarial_anual: float,
    take_rate: float,
    salario_inicial_anual: float,
    empleabilidad_tabla: dict,
    pago_total_escuela: float,                                   # ya calculado fuera
) -> pd.DataFrame:
    curva_emp = generar_curva_empleabilidad(categoria_key, meses=36, default_rate=default_rate, tabla=empleabilidad_tabla)
    meses_empleo = asignar_mes_empleo(curva_emp, num_estudiantes)

    max_meses = 120
    cf_escuela_fijo = np.zeros(max_meses)
    cf_escuela_variable = np.zeros(max_meses)
    cf_origination = np.zeros(max_meses)
    cf_inflow_est = np.zeros(max_meses)

    # Pagos fijos a escuela (agregados por N estudiantes)
    for monto_por_est, mes in pagos_escuela_fijos_euros_por_est:
        if 1 <= mes <= max_meses:
            cf_escuela_fijo[mes - 1] += monto_por_est * num_estudiantes

    # Origination fee (mes 1) por N estudiantes
    cf_origination[0] = origination_fee * importe_financiado * num_estudiantes

    # Por estudiante: variable a escuela + inflows de repago
    for mes_emp in meses_empleo:
        var, infl = simular_estudiante(
            mes_empleo=mes_emp,
            cap_individual=cap_individual,
            duracion_curso=duracion_curso,
            salario_inicial_anual=salario_inicial_anual,
            subida_salarial_anual=subida_salarial_anual,
            take_rate=take_rate,
            pago_total_escuela=pago_total_escuela,
            pagos_escuela_fijos_pct=pagos_escuela_fijos_pct,
        )
        cf_escuela_variable += var
        cf_inflow_est += infl

    df = pd.DataFrame({
        'Mes': range(1, max_meses + 1),
        'Outflow_Escuela_Fijo': -cf_escuela_fijo,
        'Outflow_Escuela_Variable': -cf_escuela_variable,
        'Outflow_Origination_Fee': -cf_origination,
        'Inflow_Estudiantes': cf_inflow_est,
    })
    df['Cashflow_Neto'] = df[['Outflow_Escuela_Fijo','Outflow_Escuela_Variable','Outflow_Origination_Fee']].sum(axis=1) + df['Inflow_Estudiantes']
    return df


def tir_anual_desde_flujos(cashflows: np.ndarray) -> float:
    try:
        tirm = irr(cashflows)
        return (1 + tirm)**12 - 1
    except Exception:
        return np.nan

def encontrar_cap_optimo(func_tir, cap_min: float, cap_max: float, tir_objetivo: float) -> float:
    def objetivo(cap):
        t = func_tir(cap)
        if np.isnan(t):
            return 1e9
        return abs(t - tir_objetivo)
    res = minimize_scalar(objetivo, bounds=(cap_min, cap_max), method='bounded')
    return float(res.x)

def tir_amortizacion(
    mes_abs: int,
    cap_amort: float,
    importe_financiado: float,
    pago_total_escuela: float,
    origination_fee: float,
    pagos_fijos_pct: List[Tuple[float, int]],   # porcentajes (mes 1 incluye concesión)
) -> float:
    cash = np.zeros(mes_abs)
    # Meses con pagos fijos a escuela (en % sobre pago_total_escuela)
    for pct, mes in pagos_fijos_pct:
        if 1 <= mes <= len(cash):
            cash[mes-1] -= pct * pago_total_escuela
    # Origination (mes 1)
    cash[0] -= origination_fee * importe_financiado
    # Mes ventana: entra CAP amortización y sale residual a escuela
    suma_fijos = sum(p for p,_ in pagos_fijos_pct)
    residual = max(0.0, 1 - suma_fijos)
    cash[mes_abs-1] += cap_amort
    cash[mes_abs-1] -= residual * pago_total_escuela
    return tir_anual_desde_flujos(cash)

def encontrar_cap_amortizacion(
    mes_abs: int,
    cap_min: float,
    cap_max: float,
    tir_objetivo: float,
    importe_financiado: float,
    pago_total_escuela: float,
    origination_fee: float,
    pagos_fijos_pct: List[Tuple[float, int]],
) -> float:
    def objetivo(cap):
        t = tir_amortizacion(mes_abs, cap, importe_financiado, pago_total_escuela, origination_fee, pagos_fijos_pct)
        if np.isnan(t):
            return 1e9
        return abs(t - tir_objetivo)
    res = minimize_scalar(objetivo, bounds=(cap_min, cap_max), method='bounded')
    return float(res.x)


# ============================================
# FastAPI
# ============================================

app = FastAPI(title="Bcas ISA Calc API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/propuesta", response_model=Proposal)
def propuesta(p: Payload):
    # 1) Normalizaciones y parámetros
    categoria_key = CATEGORIA_CENTRO_MAP[p.categoria_centro_legible]
    empleabilidad_tabla = EMPLEABILIDAD_TABLA_DEFAULT

    salario_categorias = p.salario_categorias or SALARIO_CATEGORIAS_DEFAULT
    cat_norm = _norm(p.categoria_salario)
    if cat_norm not in salario_categorias:
        raise HTTPException(status_code=422, detail={
            "msg": f"categoria_salario inválida: '{p.categoria_salario}'",
            "permitidas": list(salario_categorias.keys())
        })
    salario_inicial_anual = salario_categorias[cat_norm]

    # 2) Descuento por rangos y pago total a escuela por estudiante
    pago_total_escuela = calcular_pago_escuela_total(p.importe_financiado, p.descuento_tramos)

    # 3) Pagos fijos
    #    - porcentajes (incluyendo concesión en mes 1)
    pagos_fijos_pct = [(porc + p.pago_concesion, mes) if mes == 1 else (porc, mes) for (porc, mes) in p.pagos_escuela_fijos]
    #    - euros por estudiante (para outflow fijo agregado)
    pagos_fijos_euros_por_est = [(pct * pago_total_escuela, mes) for (pct, mes) in pagos_fijos_pct]

    # 4) Función TIR(cap) para optimizar
    NUM_EST = 100
    def tir_para_cap(cap_val: float) -> float:
        df = simular_spv(
            cap_individual=cap_val,
            categoria_key=categoria_key,
            num_estudiantes=NUM_EST,
            duracion_curso=p.duracion_curso,
            default_rate=p.default_rate,
            origination_fee=p.origination_fee,
            importe_financiado=p.importe_financiado,
            pagos_escuela_fijos_euros_por_est=pagos_fijos_euros_por_est,
            pagos_escuela_fijos_pct=pagos_fijos_pct,
            subida_salarial_anual=p.subida_salarial_anual,
            take_rate=p.take_rate,
            salario_inicial_anual=salario_inicial_anual,
            empleabilidad_tabla=empleabilidad_tabla,
            pago_total_escuela=pago_total_escuela,
        )
        cash = df['Cashflow_Neto'].to_numpy()
        return tir_anual_desde_flujos(cash)

    # 5) CAP óptimo
    cap_min, cap_max = p.importe_financiado, p.importe_financiado * 3
    cap_opt = encontrar_cap_optimo(tir_para_cap, cap_min, cap_max, p.tir_objetivo)
    cap_final = float(round(cap_opt / 10.0) * 10.0)

    # 6) Ventanas 6/12/18 post-formación
    ventanas_pf = [6, 12, 18]
    caps_ventanas = []
    cap_min_ventana = p.importe_financiado  # primera ventana al menos = importe financiado
    for pf in ventanas_pf:
        mes_abs = p.duracion_curso + pf
        cap_v = encontrar_cap_amortizacion(
            mes_abs=mes_abs,
            cap_min=cap_min_ventana,
            cap_max=p.importe_financiado * 3,
            tir_objetivo=p.tir_objetivo,
            importe_financiado=p.importe_financiado,
            pago_total_escuela=pago_total_escuela,
            origination_fee=p.origination_fee,
            pagos_fijos_pct=pagos_fijos_pct,
        )
        cap_v_round = float(round(cap_v / 10.0) * 10.0)
        caps_ventanas.append(cap_v_round)
        # Escalera mínima 4% vs ventana anterior
        cap_min_ventana = cap_v_round * 1.04

    res = ISAResult(
        cap_final=cap_final,
        ventana1=caps_ventanas[0],
        ventana2=caps_ventanas[1],
        ventana3=caps_ventanas[2],
    )
    return Proposal(isa=res)
