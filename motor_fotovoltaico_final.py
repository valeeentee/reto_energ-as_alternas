import streamlit as st
import pandas as pd
import numpy as np
import pvlib
from pvlib import irradiance
from datetime import datetime, timedelta
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium

# ==================================================================
# CONFIGURACIÓN DE PÁGINA ANCHA (Debe ser la primera directiva)
# ==================================================================
st.set_page_config(layout="wide", page_title="Motor GDMTH Avanzado")

st.title("🔧 Motor GDMTH - Analizador Geográfico y Temporal Pro")
st.markdown("Mueve las coordenadas escribiendo o dando clic directo en el mapa para recalcular el motor fotovoltaico.")

# Tarifa fija por defecto de CFE si no se carga archivo de tarifas
TARIFAS_DEFAULT = {'Base': 1.0148, 'Inter': 1.5744, 'Punta': 1.7084, 'Dist': 57.74, 'Capa': 398.22, 'Fijo': 502.03}

# ==================================================================
# INITIALIZE SESSION STATE PARA COORDENADAS (Default: Tec de Monterrey)
# ==================================================================
if 'lat' not in st.session_state:
    st.session_state.lat = 25.6514
if 'lon' not in st.session_state:
    st.session_state.lon = -100.2905

# ==================================================================
# 1. ENTRADAS DE COORDENADAS (MANUALES)
# ==================================================================
st.subheader("📍 1. Ubicación del Proyecto")
col_coords = st.columns(2)

with col_coords[0]:
    nueva_lat = st.number_input("Latitud", value=st.session_state.lat, format="%.4f", key="lat_input")
    if nueva_lat != st.session_state.lat:
        st.session_state.lat = nueva_lat
        st.rerun()

with col_coords[1]:
    nueva_lon = st.number_input("Longitud", value=st.session_state.lon, format="%.4f", key="lon_input")
    if nueva_lon != st.session_state.lon:
        st.session_state.lon = nueva_lon
        st.rerun()

# ==================================================================
# MAPA INTERACTIVO (Sincronizado de dos vías - Vista Satelital Google)
# ==================================================================
m = folium.Map(
    location=[st.session_state.lat, st.session_state.lon], 
    zoom_start=15, # Un poco más de zoom para apreciar el campus
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", 
    attr="Google Satélite"
)
folium.Marker([st.session_state.lat, st.session_state.lon], popup="Ubicación seleccionada", icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
m.add_child(folium.LatLngPopup()) 

map_data = st_folium(m, height=350, width=1200, key="mapa_proyecto")

if map_data and map_data.get("last_clicked"):
    clic_lat = map_data["last_clicked"]["lat"]
    clic_lon = map_data["last_clicked"]["lng"]
    if clic_lat != st.session_state.lat or clic_lon != st.session_state.lon:
        st.session_state.lat = clic_lat
        st.session_state.lon = clic_lon
        st.rerun()

st.success(f"📍 Coordenadas Activas -> Latitud: {st.session_state.lat:.4f} | Longitud: {st.session_state.lon:.4f}")

# ==================================================================
# 2. SIDEBAR - CONFIGURACIÓN TÉCNICA Y DE ENTRADA
# ==================================================================
with st.sidebar:
    st.header("📂 Configuración de Demanda")
    max_demand = st.selectbox("Demanda Máxima Planta (kW)", [30, 50, 60], index=1)
    plant_factor = st.slider("Factor de Planta", 0.50, 0.70, 0.60, 0.01)

    st.header("☀️ Datos del Panel")
    pnom = st.number_input("Potencia Panel (Wp)", 450)
    num_paneles = st.number_input("Número de paneles", 50, step=1)
    pnom_total = pnom * num_paneles
    tilt_optimo = st.slider("Inclinación Fija (°)", 0, 90, 15)
    alt = st.number_input("Altitud (m)", 540) # Altitud promedio de Monterrey aprox 540m

    st.header("📅 Rango Temporal a Evaluar")
    fecha_inicio = st.date_input("Fecha de inicio", datetime(2024, 1, 1))
    fecha_fin = st.date_input("Fecha de fin", datetime(2024, 12, 31))

# ==================================================================
# 3. LÓGICA MATEMÁTICA Y SIMULACIÓN (MOTOR JENSEN / INEICHEN)
# ==================================================================
def generar_demanda_sintetica(max_demand_kw, plant_factor):
    idx = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:45:00', freq='15min')
    
    # Extraemos horas, meses y días como arreglos de numpy puros (.values)
    horas_float = idx.hour.values + idx.minute.values / 60.0
    meses = idx.month.values
    dias_semana = idx.dayofweek.values
    
    # Perfil diurno y factores estacionales estables
    diurno = np.clip(0.6 + 0.4 * np.sin(np.pi * (horas_float - 6) / 12), 0.4, 1.0)
    factor_estacion = np.where((meses >= 6) & (meses <= 9), 1.2, 1.0)
    factor_fin_semana = np.where(dias_semana >= 5, 0.6, 1.0)
    
    perfil_base = float(max_demand_kw) * diurno * factor_estacion * factor_fin_semana
    
    # Forzar el factor de planta exacto solicitado
    escala = plant_factor / (perfil_base.mean() / max_demand_kw)
    perfil_final = perfil_base * escala
    
    return pd.DataFrame({'demanda_kw': perfil_final}, index=idx)

def calcular_generacion_anual_ineichen(lat, lon, alt, tilt, azimut, pnom_w):
    times = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:45:00', freq='15min')
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    
    clearsky = pvlib.clearsky.ineichen(
        solpos['apparent_zenith'].values, 
        airmass_absolute=1.5, 
        linke_turbidity=3, 
        altitude=alt
    )
    
    poa = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimut,
        solar_zenith=solpos['apparent_zenith'].values,
        solar_azimuth=solpos['azimuth'].values,
        dni=np.asarray(clearsky['dni']).ravel(),
        ghi=np.asarray(clearsky['ghi']).ravel(),
        dhi=np.asarray(clearsky['dhi']).ravel()
    )
    
    gen = np.clip(np.nan_to_num((poa['poa_global'] / 1000) * (pnom_w / 1000) * 0.96 * 0.85, nan=0.0), 0, None)
    return pd.DataFrame({'generacion_kw': gen, 'poa_wm2': poa['poa_global']}, index=times)

# ==================================================================
# 4. EJECUCIÓN CONTINUA ANUALIZADA
# ==================================================================
df_demanda = generar_demanda_sintetica(max_demand, plant_factor)

df_gen_directo = calcular_generacion_anual_ineichen(st.session_state.lat, st.session_state.lon, alt, tilt_optimo, 183, pnom_total)
df_gen_punta = calcular_generacion_anual_ineichen(st.session_state.lat, st.session_state.lon, alt, tilt_optimo, 270, pnom_total)

df_res_dir = pd.concat([df_demanda, df_gen_directo], axis=1).fillna(0)
df_res_pun = pd.concat([df_demanda, df_gen_punta], axis=1).fillna(0)

# ==================================================================
# 5. RECORTE TEMPORAL (Filtro por fechas)
# ==================================================================
t_start = pd.to_datetime(fecha_inicio)
t_end = pd.to_datetime(fecha_fin) + timedelta(days=1) - timedelta(minutes=15)

df_filtrado_dir = df_res_dir.loc[t_start:t_end]
df_filtrado_pun = df_res_pun.loc[t_start:t_end]

if df_filtrado_dir.empty:
    st.warning("⚠️ El rango de fechas seleccionado está fuera de la serie temporal disponible.")
    st.stop()

# ==================================================================
# 6. DESPLIEGUE DE RESULTADOS EN PANTALLA
# ==================================================================
st.markdown(f"## 📊 Análisis para el Periodo: {fecha_inicio} al {fecha_fin}")

c_res1, c_res2 = st.columns(2)

with c_res1:
    st.markdown("### ☀️ Escenario: PV Directo (Sur - 183°)")
    st.metric("Generación en este periodo", f"{(df_filtrado_dir['generacion_kw'].sum()*0.25):,.1f} kWh")

with c_res2:
    st.markdown("### 🌇 Escenario: PV Punta (Oeste - 270°)")
    st.metric("Generación en este periodo", f"{(df_filtrado_pun['generacion_kw'].sum()*0.25):,.1f} kWh")

if (t_end - t_start).days > 14:
    df_p_dir = df_filtrado_dir.resample('D').mean()
    df_p_pun = df_filtrado_pun.resample('D').mean()
    title_grafica = "Comportamiento Promedio Diario (Periodo Amplio)"
else:
    df_p_dir = df_filtrado_dir
    df_p_pun = df_filtrado_pun
    title_grafica = "Comportamiento Quinceminutal Dinámico"

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_p_dir.index, y=df_p_dir['demanda_kw'], name='Demanda Planta', line=dict(color='black', width=2)))
fig.add_trace(go.Scatter(x=df_p_dir.index, y=df_p_dir['generacion_kw'], name='Gen PV Directo (Sur)', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df_p_pun.index, y=df_p_pun['generacion_kw'], name='Gen PV Punta (Oeste)', line=dict(color='red', dash='dash')))
fig.update_layout(title=title_grafica, xaxis_title="Tiempo", yaxis_title="Potencia (kW)", legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# 7. EXPORTACIÓN DE ENTREGABLES
# ==================================================================
st.subheader("📥 Descargar Datos del Periodo Seleccionado")
col_d1, col_d2 = st.columns(2)
col_d1.download_button("Descargar Filtro - PV Directo (CSV)", df_filtrado_dir.to_csv(), "Filtro_PV_Directo.csv", "text/csv")
col_d2.download_button("Descargar Filtro - PV Punta (CSV)", df_filtrado_pun.to_csv(), "Filtro_PV_Punta.csv", "text/csv")