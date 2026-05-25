import streamlit as st
import pandas as pd
import numpy as np
import pvlib
from pvlib import irradiance, atmosphere, iotools
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# 1. FUNCIÓN PARA GENERAR DEMANDA SINTÉTICA (15 min)
# ------------------------------------------------------------
def generar_demanda_15min(max_demand_kw, plant_factor, power_factor, anio=2024):
    """
    Genera perfil de demanda cada 15 min para un año.
    """
    start = datetime(anio, 1, 1, 0, 0)
    end = datetime(anio, 12, 31, 23, 45)
    idx = pd.date_range(start=start, end=end, freq='15min')
    
    hora = idx.hour + idx.minute/60.0
    diurno = 0.6 + 0.4 * np.sin(np.pi * (hora - 6) / 12)
    diurno = np.clip(diurno, 0.4, 1.0)
    
    es_verano = (idx.month >= 6) & (idx.month <= 9)
    factor_estacion = np.where(es_verano, 1.2, 1.0)
    
    es_fin_semana = (idx.dayofweek >= 5)
    factor_fin_semana = np.where(es_fin_semana, 0.6, 1.0)
    
    perfil = max_demand_kw * diurno * factor_estacion * factor_fin_semana
    perfil = pd.Series(perfil, index=idx)  # Asegurar que sea Series
    
    media_calculada = perfil.mean()
    factor_planta_calculado = media_calculada / max_demand_kw
    perfil = perfil * (plant_factor / factor_planta_calculado)
    perfil = np.clip(perfil, 0, max_demand_kw)
    
    df_demanda = pd.DataFrame({'demanda_kw': perfil}, index=idx)
    df_demanda['power_factor'] = power_factor
    return df_demanda

# ------------------------------------------------------------
# 2. OBTENER DATOS METEOROLÓGICOS (TMY horario)
# ------------------------------------------------------------
@st.cache_data(ttl=86400)
def obtener_irradiancia_tmy(lat, lon, altitud, anio=2024):
    try:
        data, meta = iotools.get_pvgis_tmy(
            latitude=lat, longitude=lon,
            outputformat='csv', usehorizon=True, userhorizon=None,
            startyear=None, endyear=None, map_variables=True
        )
        data.index = pd.date_range(start=f'{anio}-01-01 00:00:00', 
                                   periods=len(data), freq='h')
        return data
    except Exception as e:
        st.warning(f"Error descargando TMY: {e}. Usando modelo de cielo claro.")
        times = pd.date_range(start=f'{anio}-01-01 00:00:00', 
                              end=f'{anio}-12-31 23:00:00', freq='h')
        solpos = pvlib.solarposition.get_solarposition(times, lat, lon, altitud)
        pressure = atmosphere.alt2pres(altitud)
        ghi_clear = pvlib.clearsky.ineichen(times, lat, lon, altitud, pressure=pressure)
        dhi_clear = pvlib.clearsky.dhi_ineichen(times, ghi_clear, solpos['zenith'])
        dni_clear = pvlib.clearsky.dni_ineichen(times, ghi_clear, dhi_clear, solpos['zenith'])
        data = pd.DataFrame({
            'ghi': ghi_clear, 'dni': dni_clear, 'dhi': dhi_clear,
            'temp_air': 20.0, 'wind_speed': 1.0
        }, index=times)
        return data

# ------------------------------------------------------------
# 3. CALCULAR GENERACIÓN 15 MIN (modelo isotrópico, sin dni_extra)
# ------------------------------------------------------------
def calcular_generacion_15min(data_hora, lat, lon, altitud, tilt, azimuth,
                              area_m2, eficiencia_panel, potencia_nominal_w,
                              pr=0.85, inverter_eff=0.96):
    
    idx_15min = pd.date_range(start=data_hora.index[0], 
                              end=data_hora.index[-1] + timedelta(hours=1) - timedelta(minutes=15),
                              freq='15min')
    
    ghi_15 = data_hora['ghi'].resample('15min').interpolate(method='linear')
    dni_15 = data_hora['dni'].resample('15min').interpolate(method='linear')
    dhi_15 = data_hora['dhi'].resample('15min').interpolate(method='linear')
    
    ghi_15 = ghi_15.reindex(idx_15min)
    dni_15 = dni_15.reindex(idx_15min)
    dhi_15 = dhi_15.reindex(idx_15min)
    
    solpos = pvlib.solarposition.get_solarposition(idx_15min, lat, lon, altitud)
    
    # Modelo isotrópico - no requiere dni_extra
    poa = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos['zenith'],
        solar_azimuth=solpos['azimuth'],
        dni=dni_15,
        ghi=ghi_15,
        dhi=dhi_15,
        model='isotropic'
    )
    poa_global = poa['poa_global']
    
    generacion_dc_kw = (poa_global / 1000.0) * (potencia_nominal_w / 1000.0)
    generacion_ac_kw = generacion_dc_kw * inverter_eff * pr
    generacion_ac_kw = generacion_ac_kw.clip(lower=0)
    
    df_15min = pd.DataFrame({
        'poa_global_wm2': poa_global,
        'generacion_kw': generacion_ac_kw
    }, index=idx_15min)
    return df_15min

# ------------------------------------------------------------
# 4. INTERFAZ STREAMLIT
# ------------------------------------------------------------
st.set_page_config(page_title="Motor Fotovoltaico - Jensen", layout="wide")
st.title("🔧 Motor de Generación Fotovoltaica (Modelo de Transposición Isótropo)")
st.markdown("Simulación de irradiancia POA y generación eléctrica cada 15 min para análisis GDMTH")

with st.sidebar:
    st.header("📍 Ubicación")
    lat = st.number_input("Latitud", value=20.0, format="%.4f")
    lon = st.number_input("Longitud", value=-100.0, format="%.4f")
    alt = st.number_input("Altitud (m)", value=1500)
    
    st.header("📐 Geometría de instalación")
    tilt = st.slider("Inclinación (Tilt) °", 0, 90, 20)
    azimuth = st.slider("Azimut (0=Sur, 90=Oeste, -90=Este)", -180, 180, 0)
    
    st.header("⚡ Especificaciones del panel comercial")
    col1, col2 = st.columns(2)
    with col1:
        potencia_nominal_w = st.number_input("Potencia nominal (Wp)", min_value=100, max_value=800, value=450)
        eficiencia = st.number_input("Eficiencia (%)", min_value=10.0, max_value=25.0, value=20.0) / 100.0
    with col2:
        area_m2 = st.number_input("Área (m²)", min_value=0.5, max_value=3.0, value=2.0)
    
    st.header("🔧 Parámetros de sistema")
    performance_ratio = st.slider("Performance Ratio (PR)", 0.70, 0.90, 0.85, 0.01)
    inverter_eff = st.slider("Eficiencia inversor", 0.92, 0.99, 0.96, 0.01)
    
    st.header("🏭 Demanda industrial sintética")
    max_dem = st.selectbox("Demanda máxima (kW)", [30, 50, 60], index=1)
    plant_factor = st.slider("Factor de planta", 0.50, 0.70, 0.60)
    pf = st.slider("Factor de potencia", 0.70, 0.95, 0.85)
    anio_sim = st.number_input("Año simulación", value=2024, min_value=2020, max_value=2030)

if st.sidebar.button("🚀 Ejecutar simulación", type="primary"):
    with st.spinner("Descargando datos meteorológicos..."):
        data_hora = obtener_irradiancia_tmy(lat, lon, alt, anio_sim)
    
    with st.spinner("Calculando POA y generación (15 min)..."):
        df_gen = calcular_generacion_15min(data_hora, lat, lon, alt, tilt, azimuth,
                                           area_m2, eficiencia, potencia_nominal_w,
                                           pr=performance_ratio, inverter_eff=inverter_eff)
    
    with st.spinner("Generando perfil de demanda..."):
        df_demanda = generar_demanda_15min(max_dem, plant_factor, pf, anio_sim)
    
    df_completo = pd.concat([df_gen, df_demanda], axis=1)
    df_completo['energia_generada_kwh'] = df_completo['generacion_kw'] * (15/60)
    df_completo['energia_demandada_kwh'] = df_completo['demanda_kw'] * (15/60)
    df_completo['balance_kw'] = df_completo['generacion_kw'] - df_completo['demanda_kw']
    
    total_generacion = df_completo['energia_generada_kwh'].sum()
    total_demanda = df_completo['energia_demandada_kwh'].sum()
    cobertura = total_generacion / total_demanda * 100 if total_demanda > 0 else 0
    
    st.success(f"✅ Energía generada anual: **{total_generacion:,.0f} kWh**")
    st.info(f"Demanda total anual: {total_demanda:,.0f} kWh | Cobertura solar: {cobertura:.1f}%")
    
    # Gráfica de una semana
    semana_inicio = datetime(anio_sim, 7, 1)
    df_semana = df_completo.loc[semana_inicio:semana_inicio+timedelta(days=7)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_semana.index, y=df_semana['generacion_kw'], name='Generación PV (kW)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_semana.index, y=df_semana['demanda_kw'], name='Demanda (kW)', line=dict(color='blue')))
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Potencia (kW)", title="Perfil 15 min - Julio (semana tipo)")
    st.plotly_chart(fig, use_container_width=True)
    
    # POA diario medio
    df_poa_dia = df_gen['poa_global_wm2'].resample('D').mean()
    fig2 = px.line(x=df_poa_dia.index, y=df_poa_dia, labels={'x':'Día', 'y':'POA medio (W/m²)'})
    st.plotly_chart(fig2, use_container_width=True)
    
    # Descarga CSV
    cols = ['poa_global_wm2', 'generacion_kw', 'demanda_kw', 'energia_generada_kwh', 'energia_demandada_kwh', 'balance_kw']
    csv = df_completo[cols].to_csv(encoding='utf-8-sig')
    st.download_button("📥 Descargar serie 15min (CSV)", data=csv, file_name="generacion_demanda_15min.csv", mime="text/csv")
    
    st.markdown("---")
    st.subheader("🔗 Integración con archivo Excel de apoyo")
    st.markdown("Los datos del CSV pueden importarse a `Calculos_GDMTH.xlsx` para el análisis económico.")
else:
    st.info("👈 Configura los parámetros y presiona **Ejecutar simulación**.")