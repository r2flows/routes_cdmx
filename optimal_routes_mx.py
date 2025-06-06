import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import math
import traceback
import re
import folium
from streamlit_folium import st_folium
import time
import datetime
from datetime import datetime, timedelta
import random

# Configuración de la página
st.set_page_config(
    page_title="Rutas Óptimas para Puntos por Zona",
    page_icon="🔥",
    layout="wide"
)

# Función para cargar datos CORREGIDA
@st.cache_data
def cargar_datos():
    try:
        # Cargar los archivos
        df_combinado = pd.read_csv('mx.csv', low_memory=False)
        df_cdmx_leads = pd.read_csv('cdmx_sin_duplicados.csv')
        filtros_finales = df_cdmx_leads['id'] != 18192325482
        df_cdmx_leads = df_cdmx_leads[filtros_finales].copy()
        #st.write(df_combinado.count())
        #st.write(df_cdmx_leads)
        # Preparar df_cdmx con columnas específicas
        #df_cdmx = df_cdmx_leads[['lat', 'lng', 'id', 'name', 'address']].copy()
        
        # Hacer el merge especificando sufijos para evitar confusión
        df_downtown = pd.merge(
            df_cdmx_leads, 
            df_combinado, 
            left_on='id', 
            right_on='ID de registro', 
            #on=['lat','lng'],
            how='left',
            suffixes=('_leads', '_mx')
        )
        #st.write(df_downtown.count())
        # Ahora seleccionamos las columnas que necesitamos, priorizando las del archivo leads
        # y complementando con las del archivo mx cuando sea necesario
        
        # Columnas finales que necesitamos
        df_final = pd.DataFrame()
        
        # ID de registro (usar el de leads)
        df_final['ID de registro'] = df_downtown['id']
        
        # Coordenadas (usar las de leads)
        df_final['lat'] = df_downtown['lat_leads']
        df_final['lng'] = df_downtown['lng_leads']
        
        # Nombre unificado (priorizar leads)
        df_final['nombre_unificado'] = df_downtown['name_leads'].fillna(df_downtown.get('Nombre de la empresa', ''))
        
        # Dirección unificada (usar address de leads)
        df_final['direccion_unificada'] = df_downtown['address']
        df_final['stage_name']= df_downtown['stage_name']
        # Agregar otras columnas importantes del archivo mx si existen
        columnas_mx_importantes = [
            'Propietario del registro de empresa',
            'Estado del lead',
            'Etapa del ciclo de vida',
            'Número de veces contactado',
            'frecuencia_compra',
            'cantidad_de_sucursales'
        ]
        
        for col in columnas_mx_importantes:
            if col in df_downtown.columns:
                df_final[col] = df_downtown[col]
        
        # Convertir coordenadas a numérico
        df_final['lat'] = pd.to_numeric(df_final['lat'], errors='coerce')
        df_final['lng'] = pd.to_numeric(df_final['lng'], errors='coerce')
        
        # Eliminar filas sin coordenadas válidas
        df_final = df_final.dropna(subset=['lat', 'lng'])
        
        # Filtrar solo registros que no sean clientes activos para prospección
        #if 'stage_name' in df_final.columns:
            # Opcional: puedes descomentar la siguiente línea si quieres excluir clientes activos
            #df_final = df_final[~df_final['stage_name'].isin(['Active clients', 'Active WinBack'])]
        #    pass
        
        st.success(f"Datos cargados exitosamente: {len(df_final)} registros")
        #st.write(df_final.count())
        return df_final
    
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        st.error("Detalles del error:")
        st.code(traceback.format_exc())
        return pd.DataFrame()

# Función para asignar ponderaciones (sin cambios)
def asignar_ponderaciones(df):
    try:
        # Diccionario de ponderaciones
        ponderaciones = {
            'Active clients': 1.0,
            'Active WinBack': 1.0,
            'Waiting 4 Payment': 0.9,
            'ReNegotiation': 0.8,
            'Negociación': 0.8,
            'ReActivated': 0.7,
            'Trials': 0.6,
            'First purchase': 0.5,
            'Potential Return': 0.5,
            'Zombies': 0.4,
            'Reloaded Zombies': 0.4,
            'Onboarding': 0.3,
            'Potential interest': 0.3,
            'Churn': 0.2,
            'DoubleChurn': 0.2,
            'Lost': 0.1
        }
        
        # Aplicar ponderaciones
        if 'stage_name' in df.columns:
            df['ponderacion'] = df['stage_name'].map(ponderaciones).fillna(0.1)
        else:
            df['ponderacion'] = 0.1
            
        return df
    except Exception as e:
        st.error(f"Error al asignar ponderaciones: {str(e)}")
        return df

# Función para calcular la distancia en km entre coordenadas (sin cambios)
def calcular_distancia_km(lat1, lon1, lat2, lon2):
    try:
        # Radio de la tierra en kilómetros
        R = 6371.0
        
        # Convertir grados a radianes
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Diferencias
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        # Fórmula de Haversine
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        # Distancia
        distance = R * c
        
        return distance
    except:
        # Fallback simple si hay error
        return ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5 * 111.0

# Función para crear hot-spots en downtown CDMX (sin cambios)
@st.cache_data
def crear_hotspots_downtown(df, radio_max_km=2.0, min_puntos=5, max_zonas=5):
    try:
        if df.empty or 'lat' not in df.columns or 'lng' not in df.columns:
            return None, None
        
        # Limpiar y preparar datos
        df_clean = df.dropna(subset=['lat', 'lng']).copy()
        df_clean['lat'] = pd.to_numeric(df_clean['lat'], errors='coerce')
        df_clean['lng'] = pd.to_numeric(df_clean['lng'], errors='coerce')
        df_clean = df_clean.dropna(subset=['lat', 'lng'])
        
        if len(df_clean) < min_puntos:
            return None, None
        
        # Centro aproximado de CDMX (Zócalo)
        centro_cdmx_lat = 19.432608
        centro_cdmx_lng = -99.133209
        
        # Calcular distancia al centro para cada punto
        df_clean['dist_centro'] = df_clean.apply(
            lambda row: calcular_distancia_km(row['lat'], row['lng'], centro_cdmx_lat, centro_cdmx_lng),
            axis=1
        )
        
        # Ordenar por distancia al centro
        df_clean = df_clean.sort_values('dist_centro')
        
        # Tomar puntos más cercanos al centro
        n_downtown = min(len(df_clean), 1000)  # Limitar a 1000 puntos para rendimiento
        df_downtown = df_clean.head(n_downtown).copy()
        
        # Método K-means para identificar clusters principales
        n_clusters = max(5, min(10, len(df_downtown) // 30))  # Entre 5 y 10 clusters según datos
        
        coords = df_downtown[['lat', 'lng']].values
        
        # Aplicar K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        df_downtown['kmeans_cluster'] = kmeans.fit_predict(coords)
        
        # Procesar cada cluster de kmeans
        cluster_stats = []
        
        for km_id in range(n_clusters):
            km_points = df_downtown[df_downtown['kmeans_cluster'] == km_id]
            
            if len(km_points) >= min_puntos:
                # Centro
                centro_lat = km_points['lat'].mean()
                centro_lng = km_points['lng'].mean()
                
                # Radio
                max_dist_km = 0
                for _, point in km_points.iterrows():
                    dist = calcular_distancia_km(centro_lat, centro_lng, point['lat'], point['lng'])
                    max_dist_km = max(max_dist_km, dist)
                
                # Solo procesar clusters con radio dentro del máximo permitido
                if max_dist_km <= radio_max_km:
                    # Calcular métricas
                    potencial_medio = km_points['ponderacion'].mean() if 'ponderacion' in km_points.columns else 0.0
                    
                    # Verificar clientes
                    tiene_pos = km_points['pos id'].notna().sum() if 'pos id' in km_points.columns else 0
                    
                    es_cliente = 0
                    if 'stage_name' in km_points.columns:
                        es_cliente += km_points[km_points['stage_name'] == 'Active clients'].shape[0]
                        es_cliente += km_points[km_points['stage_name'] == 'Active WinBack'].shape[0]
                    
                    # Porcentaje de clientes
                    porcentaje_clientes = (tiene_pos + es_cliente) / len(km_points) if len(km_points) > 0 else 0
                    
                    # Determinar nombre de zona
                    nombre_zona = f"Zona Centro {km_id+1}"
                    if 'vicinity' in km_points.columns:
                        colonias = km_points['vicinity'].dropna().tolist()
                        if colonias:
                            nombre_zona = f"Zona {max(set(colonias), key=colonias.count)}"
                    
                    # Calcular densidad
                    densidad_normalizada = min(1.0, len(km_points) / max(math.pi * max_dist_km**2, 0.01) / 100)
                    
                    # Valor combinado
                    valor_combinado = (
                        potencial_medio * 0.6 + 
                        densidad_normalizada * 0.2 
                    )
                    
                    # Crear info del cluster
                    cluster_info = {
                        'cluster_id': km_id,
                        'centro_lat': centro_lat,
                        'centro_lng': centro_lng,
                        'radio_km': max_dist_km,
                        'num_puntos': len(km_points),
                        'potencial_medio': potencial_medio,
                        'densidad_normalizada': densidad_normalizada,
                        'porcentaje_clientes': porcentaje_clientes,
                        'valor_combinado': valor_combinado,
                        'nombre_zona': nombre_zona,
                        'puntos': km_points
                    }
                    
                    cluster_stats.append(cluster_info)
        
        # Ordenar y limitar
        cluster_stats.sort(key=lambda x: x['valor_combinado'], reverse=True)
        top_clusters = cluster_stats[:max_zonas]
        
        # Crear DataFrame de resultados
        resultados_df = pd.DataFrame({
            'Zona': [i+1 for i in range(len(top_clusters))],
            'Nombre': [c['nombre_zona'] for c in top_clusters],
            'Puntos': [c['num_puntos'] for c in top_clusters],
            'Radio (km)': [round(c['radio_km'], 2) for c in top_clusters],
            'Potencial': [round(c['potencial_medio'], 2) for c in top_clusters],
            'Valor': [round(c['valor_combinado'], 2) for c in top_clusters]
        })
        
        return resultados_df, top_clusters
        
    except Exception as e:
        st.error(f"Error al crear hot-spots: {str(e)}")
        st.error(traceback.format_exc())
        return None, None

# Función para visualizar hot-spots (sin cambios)
def visualizar_hotspots(cluster_stats, mostrar_puntos=True):
    try:
        if not cluster_stats:
            return None
            
        # Crear mapa base
        fig = go.Figure()
        
        # Colores para valoración
        color_scale = [
            [0, "rgb(26, 35, 126)"],   # Azul oscuro (bajo)
            [0.25, "rgb(40, 53, 147)"], # Azul
            [0.5, "rgb(46, 125, 50)"],  # Verde
            [0.75, "rgb(249, 168, 37)"], # Amarillo
            [1.0, "rgb(183, 28, 28)"]   # Rojo (alto)
        ]
        
        # Punto central del mapa
        all_lats = [zona['centro_lat'] for zona in cluster_stats]
        all_lngs = [zona['centro_lng'] for zona in cluster_stats]
        center_lat = sum(all_lats) / len(all_lats)
        center_lng = sum(all_lngs) / len(all_lngs)
        
        # Para cada zona, crear visualización
        for i, zona in enumerate(cluster_stats):
            # Color basado en valor
            valor = min(1.0, zona['valor_combinado'])
            color_idx = int(valor * 4)
            color_zona = color_scale[color_idx][1]
            
            # Marcador principal
            fig.add_trace(go.Scattermapbox(
                lat=[zona['centro_lat']],
                lon=[zona['centro_lng']],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=color_zona,
                    opacity=0.8
                ),
                text=[f"{i+1}"],
                textfont=dict(
                    size=14,
                    color='white'
                ),
                hovertemplate=(
                    f"<b>Zona {i+1}: {zona['nombre_zona']}</b><br>" +
                    f"Valor: {zona['valor_combinado']:.2f}<br>" +
                    f"Puntos: {zona['num_puntos']}<br>" +
                    f"Radio: {zona['radio_km']:.2f} km<br>" +
                    f"<extra></extra>"
                ),
                name=f"Zona {i+1}",
                showlegend=False
            ))
            
            # Círculo de área
            theta = np.linspace(0, 2*np.pi, 100)
            radio_grados = zona['radio_km'] / 111.0
            
            x = zona['centro_lng'] + radio_grados * np.cos(theta)
            y = zona['centro_lat'] + radio_grados * np.sin(theta)
            
            fig.add_trace(go.Scattermapbox(
                lat=y,
                lon=x,
                mode='lines',
                line=dict(
                    width=2,
                    color=color_zona
                ),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Mostrar puntos si está activado
            if mostrar_puntos and 'puntos' in zona:
                puntos_df = zona['puntos']
                
                # Limitar número de puntos a mostrar para mejor rendimiento
                if len(puntos_df) > 100:
                    puntos_df = puntos_df.sample(100)
                
                # Mostrar todos los puntos (verde)
                if len(puntos_df) > 0:
                    fig.add_trace(go.Scattermapbox(
                        lat=puntos_df['lat'].tolist(),
                        lon=puntos_df['lng'].tolist(),
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='rgb(0, 128, 0)',  # Verde para todos los puntos
                            opacity=0.8
                        ),
                        hoverinfo='none',
                        showlegend=False
                    ))
        
        # Marcar el centro de CDMX
        fig.add_trace(go.Scattermapbox(
            lat=[19.432608],  # Zócalo
            lon=[-99.133209],
            mode='markers',
            marker=dict(
                size=15,
                color='rgb(255, 215, 0)',  # Dorado
                opacity=0.8,
                symbol='star'
            ),
            text=["Centro CDMX"],
            hoverinfo='text',
            showlegend=False
        ))
        
        # Configurar layout
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=center_lat, lon=center_lng),
                zoom=12
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error al visualizar hot-spots: {str(e)}")
        return None

# Función para crear mapa detallado de una zona específica (sin cambios)
def crear_mapa_zona(zona):
    try:
        if 'puntos' not in zona or zona['puntos'].empty:
            return None
            
        puntos_df = zona['puntos']
        #puntos_df = puntos_df[['ID de registro','','']]
        #st.write(puntos_df.columns)
        # Limitar puntos a mostrar para mejor rendimiento
        if len(puntos_df) > 200:
            puntos_muestra = puntos_df.sample(200).copy()
            st.info(f"Mostrando muestra de 200 puntos de un total de {len(puntos_df)} para mejor rendimiento.")
        else:
            puntos_muestra = puntos_df.copy()
            
        # Crear mapa específico
        fig_zona = go.Figure()
        
        # Añadir círculo para la zona
        theta = np.linspace(0, 2*np.pi, 100)
        radio_grados = zona['radio_km'] / 111.0
        
        x = zona['centro_lng'] + radio_grados * np.cos(theta)
        y = zona['centro_lat'] + radio_grados * np.sin(theta)
        
        fig_zona.add_trace(go.Scattermapbox(
            lat=y,
            lon=x,
            mode='lines',
            line=dict(
                width=2,
                color='rgba(183, 28, 28, 0.6)'
            ),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Añadir punto central
        fig_zona.add_trace(go.Scattermapbox(
            lat=[zona['centro_lat']],
            lon=[zona['centro_lng']],
            mode='markers',
            marker=dict(
                size=15,
                color='rgb(183, 28, 28)',
                opacity=0.8
            ),
            text=[f"Centro: {zona['nombre_zona']}"],
            hoverinfo='text',
            showlegend=False
        ))
        
        # Añadir todos los puntos
        if len(puntos_muestra) > 0:
            # Textos para los puntos
            if 'nombre_unificado' in puntos_muestra.columns:
                textos_puntos = puntos_muestra['nombre_unificado'].fillna('Punto sin nombre').tolist()
            else:
                textos_puntos = [f"Punto {j+1}" for j in range(len(puntos_muestra))]
            
            fig_zona.add_trace(go.Scattermapbox(
                lat=puntos_muestra['lat'].tolist(),
                lon=puntos_muestra['lng'].tolist(),
                mode='markers',
                marker=dict(
                    size=10,
                    color='rgb(0, 128, 0)',  # Verde para todos los puntos
                    opacity=0.9,
                    symbol='circle'
                ),
                text=textos_puntos,
                hovertemplate="%{text}<extra></extra>",
                name="Puntos",
                legendgroup="Puntos",
                showlegend=True
            ))
        
        # Configurar mapa
        fig_zona.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=zona['centro_lat'], lon=zona['centro_lng']),
                zoom=14
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.7)"
            )
        )
        
        return fig_zona
        
    except Exception as e:
        st.error(f"Error al crear mapa de zona: {str(e)}")
        return None

# NUEVA FUNCIÓN: Algoritmo de aproximación rápida para TSP (sin cambios)
def vecino_mas_cercano_optimizado(puntos_df, centro_lat, centro_lng, max_puntos=50):
    """
    Implementa un algoritmo rápido de vecino más cercano para resolver 
    aproximadamente el problema del vendedor viajero.
    
    Args:
        puntos_df (DataFrame): DataFrame con los puntos
        centro_lat (float): Latitud del centro
        centro_lng (float): Longitud del centro
        max_puntos (int): Número máximo de puntos a incluir
        
    Returns:
        tuple: (indices_ruta, distancia_total)
    """
    # Verificar que hay puntos
    if len(puntos_df) == 0:
        return [], 0
    
    # Si hay un solo punto, la ruta es trivial
    if len(puntos_df) == 1:
        return [0], calcular_distancia_km(centro_lat, centro_lng, puntos_df.iloc[0]['lat'], puntos_df.iloc[0]['lng']) * 2
    
    # Limitar a max_puntos
    puntos_limitados = puntos_df.copy()
    if len(puntos_df) > max_puntos:
        # Primero seleccionar por prioridad si hay ponderación
        if 'ponderacion' in puntos_df.columns:
            puntos_limitados = puntos_df.sort_values('ponderacion', ascending=False).head(max_puntos)
        else:
            # O por distancia al centro (con preselección para optimizar)
            puntos_df['dist_centro'] = puntos_df.apply(
                lambda row: calcular_distancia_km(centro_lat, centro_lng, row['lat'], row['lng']),
                axis=1
            )
            puntos_limitados = puntos_df.sort_values('dist_centro').head(max_puntos)
    
    # Extraer coordenadas como array para cálculos más rápidos
    lats = puntos_limitados['lat'].values
    lngs = puntos_limitados['lng'].values
    n = len(lats)
    
    # Implementación rápida de vecino más cercano
    # Comenzar desde el centro
    not_visited = set(range(n))
    ruta = []
    distancia_total = 0
    
    # Posición actual (centro)
    cur_lat, cur_lng = centro_lat, centro_lng
    
    # Construir ruta
    while not_visited:
        # Encontrar el punto más cercano
        min_dist = float('inf')
        closest = None
        
        for i in not_visited:
            dist = calcular_distancia_km(cur_lat, cur_lng, lats[i], lngs[i])
            if dist < min_dist:
                min_dist = dist
                closest = i
        
        # Agregar a la ruta
        ruta.append(closest)
        not_visited.remove(closest)
        
        # Actualizar posición actual y distancia
        distancia_total += min_dist
        cur_lat, cur_lng = lats[closest], lngs[closest]
    
    # Añadir distancia de regreso al centro
    distancia_total += calcular_distancia_km(cur_lat, cur_lng, centro_lat, centro_lng)
    
    return ruta, distancia_total

# Función auxiliar para calcular la distancia total de una ruta (sin cambios)
def calcular_distancia_total_ruta(puntos_df, centro_lat, centro_lng):
    """Calcula la distancia total aproximada de una ruta"""
    if len(puntos_df) == 0:
        return 0
    
    # Calcular desde el centro al primer punto
    distancia = calcular_distancia_km(
        centro_lat, centro_lng,
        puntos_df.iloc[0]['lat'], puntos_df.iloc[0]['lng']
    )
    
    # Sumar distancias entre puntos consecutivos
    for i in range(len(puntos_df)-1):
        distancia += calcular_distancia_km(
            puntos_df.iloc[i]['lat'], puntos_df.iloc[i]['lng'],
            puntos_df.iloc[i+1]['lat'], puntos_df.iloc[i+1]['lng']
        )
    
    # Sumar distancia del último punto al centro
    distancia += calcular_distancia_km(
        puntos_df.iloc[-1]['lat'], puntos_df.iloc[-1]['lng'],
        centro_lat, centro_lng
    )
    
    return distancia

# NUEVA FUNCIÓN MEJORADA: Verifica si una hora está en el tiempo de colación (sin cambios)
def es_hora_colacion(tiempo_dt):
    """
    Verifica si un tiempo dado está dentro del horario de colación (13:00-14:00)
    
    Args:
        tiempo_dt (datetime): Tiempo a verificar
        
    Returns:
        bool: True si está en hora de colación, False en caso contrario
    """
    return tiempo_dt.hour == 13

# NUEVA FUNCIÓN MEJORADA: Obtiene el siguiente día laboral (sin cambios)
def siguiente_dia_laboral(fecha_actual, dias_habiles=[0, 1, 2, 3, 4]):  # Por defecto Lun-Vie (0=Lunes, 6=Domingo)
    """
    Retorna la fecha del siguiente día laboral
    
    Args:
        fecha_actual (datetime.date): Fecha actual
        dias_habiles (list): Lista de días hábiles (0=Lunes, 1=Martes, ..., 6=Domingo)
        
    Returns:
        datetime.date: Fecha del siguiente día laboral
    """
    siguiente_fecha = fecha_actual + timedelta(days=1)
    
    # Buscar el siguiente día que sea laboral
    while siguiente_fecha.weekday() not in dias_habiles:
        siguiente_fecha += timedelta(days=1)
    
    return siguiente_fecha

# FUNCIÓN CLAVE COMPLETAMENTE REESCRITA: Crear itinerario con días laborales y horario (sin cambios)
def crear_itinerario_laboral(puntos_df, indices_ruta, centro_lat, centro_lng, 
                            duracion_reunion=30, tiempo_desplazamiento=10,
                            hora_inicio="09:00", hora_fin="18:00",
                            fecha_inicio=None, dias_habiles=[0, 1, 2, 3, 4]):
    """
    Crea un itinerario que incluye horarios estimados para cada visita, respetando
    horario laboral, tiempo de colación y días laborales.
    
    Args:
        puntos_df (DataFrame): DataFrame con los puntos
        indices_ruta (list): Índices de la ruta optimizada
        centro_lat (float): Latitud del centro
        centro_lng (float): Longitud del centro
        duracion_reunion (int): Duración de cada reunión en minutos
        tiempo_desplazamiento (int): Tiempo promedio de desplazamiento en minutos
        hora_inicio (str): Hora de inicio en formato "HH:MM"
        hora_fin (str): Hora de fin en formato "HH:MM"
        fecha_inicio (datetime.date): Fecha de inicio, por defecto hoy
        dias_habiles (list): Lista de días hábiles (0=Lunes, 1=Martes, ..., 6=Domingo)
        
    Returns:
        tuple: (DataFrame: Itinerario con horarios y fechas, int: Número total de días)
    """
    if not indices_ruta or len(indices_ruta) == 0:
        return pd.DataFrame(), 0
    
    # Establecer fecha de inicio si no se proporciona
    if fecha_inicio is None:
        fecha_inicio = datetime.now().date()
    elif isinstance(fecha_inicio, str):
        fecha_inicio = datetime.strptime(fecha_inicio, "%Y-%m-%d").date()
    
    # Asegurarse que la fecha de inicio sea un día laboral
    while fecha_inicio.weekday() not in dias_habiles:
        fecha_inicio += timedelta(days=1)
    
    # Obtener puntos en el orden de la ruta
    puntos_ruta = puntos_df.iloc[indices_ruta].reset_index(drop=True).copy()
    
    # Convertir horas a objetos time para comparaciones
    hora_inicio_dt = datetime.strptime(hora_inicio, "%H:%M").time()
    hora_fin_dt = datetime.strptime(hora_fin, "%H:%M").time()
    
    # Preparar listas para almacenar resultados
    fechas = []  # Fecha de cada visita
    dias_semana = []  # Día de la semana (texto)
    numero_dias = []  # Número de día en la secuencia
    ordenes_dia = []  # Orden dentro de cada día
    horarios_llegada = []  # Hora de llegada
    horarios_salida = []  # Hora de salida
    
    # Variables para seguimiento
    fecha_actual = fecha_inicio
    tiempo_actual = datetime.combine(fecha_actual, hora_inicio_dt)
    dia_actual = 1
    orden_en_dia = 1
    
    # Para cada punto en la ruta
    for i, punto in puntos_ruta.iterrows():
        # Si estamos comenzando un nuevo día, siempre empezamos a la hora de inicio
        if orden_en_dia == 1:
            tiempo_actual = datetime.combine(fecha_actual, hora_inicio_dt)
            
            # Agregar tiempo para llegar al primer punto desde el centro
            tiempo_actual += timedelta(minutes=tiempo_desplazamiento)
        
        # Verificar si estamos en hora de colación
        if es_hora_colacion(tiempo_actual):
            # Saltar a las 14:00
            tiempo_actual = datetime.combine(fecha_actual, datetime.strptime("14:00", "%H:%M").time())
        
        # Verificar si tenemos suficiente tiempo para completar la reunión antes del fin de la jornada
        tiempo_fin_reunion = tiempo_actual + timedelta(minutes=duracion_reunion)
        if tiempo_fin_reunion.time() > hora_fin_dt:
            # No hay tiempo suficiente, pasar al siguiente día laboral
            fecha_actual = siguiente_dia_laboral(fecha_actual, dias_habiles)
            tiempo_actual = datetime.combine(fecha_actual, hora_inicio_dt)
            dia_actual += 1
            orden_en_dia = 1
            
            # Agregar tiempo para llegar al primer punto desde el centro
            tiempo_actual += timedelta(minutes=tiempo_desplazamiento)
        
        # Verificar nuevamente si estamos en hora de colación después de posibles ajustes
        if es_hora_colacion(tiempo_actual):
            tiempo_actual = datetime.combine(fecha_actual, datetime.strptime("14:00", "%H:%M").time())
        
        # Registrar datos de esta visita
        fechas.append(fecha_actual.strftime("%Y-%m-%d"))
        dias_semana.append(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][fecha_actual.weekday()])
        numero_dias.append(dia_actual)
        ordenes_dia.append(orden_en_dia)
        
        # Registrar horarios
        horarios_llegada.append(tiempo_actual.strftime("%H:%M"))
        
        # Avanzar tiempo por la duración de la reunión
        tiempo_actual += timedelta(minutes=duracion_reunion)
        horarios_salida.append(tiempo_actual.strftime("%H:%M"))
        
        # Prepararse para el siguiente punto
        if i < len(puntos_ruta) - 1:
            # Calcular tiempo de desplazamiento al siguiente punto
            tiempo_actual += timedelta(minutes=tiempo_desplazamiento)
            orden_en_dia += 1
    
    # Crear DataFrame con el itinerario
    itinerario = pd.DataFrame({
        'Indice_Original': indices_ruta,
        'Orden_Global': range(1, len(puntos_ruta) + 1),
        'Dia_Numero': numero_dias,
        'Fecha': fechas,
        'Dia_Semana': dias_semana,
        'Orden_Dia': ordenes_dia,
        'Hora_Llegada': horarios_llegada,
        'Hora_Salida': horarios_salida
    })
    
    # Añadir información de los puntos
    #st.write(puntos_ruta.columns)
    if 'nombre_unificado' in puntos_ruta.columns:
        itinerario['Nombre'] = puntos_ruta['nombre_unificado'].values
    else:
        itinerario['Nombre'] = [f"Punto {i+1}" for i in range(len(puntos_ruta))]
        
    if 'direccion_unificada' in puntos_ruta.columns:
        itinerario['Direccion'] = puntos_ruta['direccion_unificada'].values
    else:
        itinerario['Direccion'] = "Dirección no disponible"
    
    itinerario['lat'] = puntos_ruta['lat'].values
    itinerario['lng'] = puntos_ruta['lng'].values
    itinerario['ID registro'] = puntos_ruta['ID de registro'].values
    itinerario['Etapa del ciclo de vida'] = puntos_ruta['Etapa del ciclo de vida'].values
    itinerario['cantidad_de_sucursales'] = puntos_ruta['cantidad_de_sucursales'].values
    itinerario['Número de veces contactado'] = puntos_ruta['Número de veces contactado'].values
    itinerario['frecuencia_compra'] = puntos_ruta['frecuencia_compra'].values

    
    if 'ponderacion' in puntos_ruta.columns:
        itinerario['Potencial'] = puntos_ruta['ponderacion'].values
    
    #if 'stage_name' in puntos_ruta.columns:
     #   itinerario['stage_name'] = puntos_ruta['stage_name'].values
    
    return itinerario, dia_actual

# Función para crear mapa de ruta por día (sin cambios)
def crear_mapa_ruta_dia(itinerario_dia, centro_lat, centro_lng, dia_numero, fecha_str, dia_semana):
    """
    Crea un mapa folium con la ruta optimizada para un día específico.
    
    Args:
        itinerario_dia (DataFrame): DataFrame con los puntos de ese día
        centro_lat (float): Latitud del centro
        centro_lng (float): Longitud del centro
        dia_numero (int): Número de día
        fecha_str (str): Fecha en formato string
        dia_semana (str): Nombre del día de la semana
        
    Returns:
        folium.Map: Mapa con la ruta
    """
    # Crear mapa base
    mapa = folium.Map(location=[centro_lat, centro_lng], zoom_start=14)
    
    # Si no hay puntos, devolver mapa vacío
    if len(itinerario_dia) == 0:
        return mapa
    
    # Añadir marcador del centro
    folium.Marker(
        location=[centro_lat, centro_lng],
        popup=f"Centro de Zona (Inicio/Fin)<br>Día {dia_numero}: {fecha_str} ({dia_semana})",
        icon=folium.Icon(color='green', icon='home')
    ).add_to(mapa)
    
    # Construir coordenadas de la ruta
    ruta_coords = [[centro_lat, centro_lng]]
    for _, punto in itinerario_dia.iterrows():
        ruta_coords.append([punto['lat'], punto['lng']])
    ruta_coords.append([centro_lat, centro_lng])
    
    # Dibujar la ruta completa como línea
    folium.PolyLine(
        ruta_coords,
        color='blue',
        weight=3,
        opacity=0.7,
        popup=f"Ruta Día {dia_numero}: {fecha_str}"
    ).add_to(mapa)
    
    # Añadir marcadores para todos los puntos en la ruta
    for _, punto in itinerario_dia.iterrows():
        # Información para popup
        nombre = punto.get('Nombre', f'Punto {punto["Orden_Dia"]}')
        direccion = punto.get('Direccion', 'Sin dirección')
        hora = punto.get('Hora_Llegada', '')
        
        # Añadir marcador
        folium.Marker(
            location=[punto['lat'], punto['lng']],
            popup=f"Parada {punto['Orden_Dia']}: {nombre}<br>{direccion}<br>Hora: {hora}",
            icon=folium.DivIcon(html=f"""
                <div style="
                    background-color: #3186cc;
                    width: 22px;
                    height: 22px;
                    border-radius: 11px;
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                    text-align: center;
                    line-height: 22px;
                ">{punto['Orden_Dia']}</div>
            """)
        ).add_to(mapa)
    
    # Calcular distancia del día
    distancia_dia = calcular_distancia_total_ruta(
        itinerario_dia,
        centro_lat, 
        centro_lng
    )
    
    # Añadir información de la ruta
    folium.Marker(
        location=[centro_lat-0.005, centro_lng],
        icon=folium.DivIcon(html=f"""
            <div style="
                background-color: rgba(255,255,255,0.8);
                padding: 5px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
                text-align: center;
                width: 200px;
            ">
                Día {dia_numero}: {fecha_str} ({dia_semana})<br>
                Visitas: {len(itinerario_dia)}<br>
                Distancia: {distancia_dia:.1f} km
            </div>
        """)
    ).add_to(mapa)
    
    return mapa

# Función para mostrar detalle de zona con ruta optimizada por días laborales (CORREGIDA)
def mostrar_detalle_zona_con_ruta_laboral(zona, index):
    """
    Muestra el detalle de una zona con su ruta optimizada respetando horario laboral.
    
    Args:
        zona (dict): Diccionario con información de la zona
        index (int): Índice de la zona
    """
    try:
        # Verificar datos de puntos
        if 'puntos' not in zona or zona['puntos'].empty:
            st.warning("No hay detalles disponibles para esta zona.")
            return
        
        puntos_df = zona['puntos'].copy()
        
        # Filtrar clientes activos si es necesario
        if 'stage_name' in puntos_df.columns:
            # Corregir los nombres de los stages
            filtro_cliente = ~puntos_df['stage_name'].isin(['Active clients', 'Active WinBack'])
            puntos_df = puntos_df[filtro_cliente]
        
        # Métricas básicas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Puntos", zona['num_puntos'])
        with col2:
            st.metric("Radio", f"{zona['radio_km']:.2f} km")
        with col3:
            if 'ponderacion' in puntos_df.columns:
                st.metric("Potencial Medio", f"{zona['potencial_medio']:.2f}")
            else:
                st.metric("Valor", f"{zona['valor_combinado']:.2f}")
        
        # Pestañas para visualización
        tab1, tab2 = st.tabs(["Mapa de Zona", "Ruta por Días Laborales"])
        
        # Tab 1: Mapa general de la zona
        with tab1:
            # Crear mapa de la zona
            fig_zona = crear_mapa_zona(zona)
            if fig_zona:
                st.plotly_chart(fig_zona, use_container_width=True)
                
            # Tabla de puntos (con limitación para rendimiento)
            st.subheader(f"Lista de puntos en Zona {index}: {zona['nombre_zona']}")
            
            # Si hay muchos puntos, ofrecer un filtro para buscar
            if len(puntos_df) > 20:
                filtro = st.text_input(f"Filtrar puntos (nombre o dirección):", key=f"filter_zone_{index}")
                if filtro:
                    # Corregir el filtrado de puntos
                    mask_nombre = puntos_df['nombre_unificado'].fillna('').str.contains(filtro, case=False)
                    mask_direccion = puntos_df['direccion_unificada'].fillna('').str.contains(filtro, case=False)
                    puntos_filtrados = puntos_df[mask_nombre | mask_direccion]
                    
                    if len(puntos_filtrados) > 0:
                        st.write(f"Mostrando {len(puntos_filtrados)} de {len(puntos_df)} puntos que coinciden con el filtro.")
                        puntos_mostrar = puntos_filtrados
                    else:
                        st.warning("No se encontraron coincidencias. Mostrando primeros 100 puntos.")
                        puntos_mostrar = puntos_df.head(100)
                else:
                    st.write(f"Mostrando primeros 100 puntos de {len(puntos_df)} totales. Use el filtro para buscar puntos específicos.")
                    puntos_mostrar = puntos_df.head(100)
            else:
                puntos_mostrar = puntos_df
            
            # Definir solo las columnas solicitadas
            cols_mostrar = []
            
            # Verificar ID de registro
            if 'ID de registro' in puntos_df.columns:
                cols_mostrar.append('ID de registro')
            
            # Añadir columnas en orden
            columnas_deseadas = ['nombre_unificado', 'direccion_unificada', 'pos id', 
                               'Propietario del registro de empresa', 'lat', 'lng']
            
            for col in columnas_deseadas:
                if col in puntos_df.columns and col not in cols_mostrar:
                    cols_mostrar.append(col)
            
            # Ordenar por ponderación si está disponible
            if 'ponderacion' in puntos_mostrar.columns:
                puntos_mostrar = puntos_mostrar.sort_values('ponderacion', ascending=False)
            
            # Mostrar tabla
            if cols_mostrar:
                st.dataframe(
                    puntos_mostrar[cols_mostrar],
                    use_container_width=True,
                    height=300
                )
                
                # Botón de descarga de todos los puntos
                csv_data = puntos_df[cols_mostrar].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Descargar Todos los Puntos de Zona {index}",
                    data=csv_data,
                    file_name=f"puntos_zona_{index}.csv",
                    mime="text/csv"
                )
            
            # Link a Google Maps
            maps_url = f"https://www.google.com/maps/search/?api=1&query={zona['centro_lat']},{zona['centro_lng']}"
            st.markdown(f"[🗺️ Abrir zona en Google Maps]({maps_url})")
        
        # Tab 2: Ruta óptima de puntos por días laborales
        with tab2:
            st.subheader(f"Ruta óptima por días laborales para Zona {index}: {zona['nombre_zona']}")
            
            # Si hay puntos, calcular ruta óptima
            if zona['num_puntos'] > 0 and len(puntos_df) > 0:
                # Parámetros de planificación en columnas
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_puntos = st.slider(
                        "Número máximo de puntos", 
                        min_value=10, 
                        max_value=min(200, len(puntos_df)), 
                        value=min(50, len(puntos_df)),
                        step=5,
                        key=f"max_puntos_{index}"
                    )
                
                with col2:
                    duracion_reunion = st.slider(
                        "Duración reunión (min)", 
                        min_value=15, 
                        max_value=60, 
                        value=30, 
                        step=5,
                        key=f"duracion_{index}"
                    )
                
                with col3:
                    tiempo_desplazamiento = st.slider(
                        "Tiempo desplazamiento (min)", 
                        min_value=5, 
                        max_value=30, 
                        value=10, 
                        step=5,
                        key=f"desplaz_{index}"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    hora_inicio = st.text_input(
                        "Hora inicio laboral (HH:MM)", 
                        "09:00", 
                        key=f"hora_inicio_{index}"
                    )
                with col2:
                    hora_fin = st.text_input(
                        "Hora fin laboral (HH:MM)", 
                        "18:00", 
                        key=f"hora_fin_{index}"
                    )
                
                # Fecha de inicio y selección de días laborales
                col1, col2 = st.columns(2)
                with col1:
                    fecha_inicio = st.date_input(
                        "Fecha de inicio",
                        datetime.now().date(),
                        key=f"fecha_inicio_{index}"
                    )
                
                with col2:
                    st.write("Días laborales:")
                    dias_laborales = []
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.checkbox("Lunes", True, key=f"lun_{index}"):
                            dias_laborales.append(0)
                        if st.checkbox("Martes", True, key=f"mar_{index}"):
                            dias_laborales.append(1)
                        if st.checkbox("Miércoles", True, key=f"mie_{index}"):
                            dias_laborales.append(2)
                        if st.checkbox("Jueves", True, key=f"jue_{index}"):
                            dias_laborales.append(3)
                    with col_b:
                        if st.checkbox("Viernes", True, key=f"vie_{index}"):
                            dias_laborales.append(4)
                        if st.checkbox("Sábado", False, key=f"sab_{index}"):
                            dias_laborales.append(5)
                        if st.checkbox("Domingo", False, key=f"dom_{index}"):
                            dias_laborales.append(6)
                
                st.info("⏰ Tiempo de colación (almuerzo) fijo: 13:00 - 14:00")
                
                # Mostrar spinner mientras se calcula la ruta
                with st.spinner("Calculando ruta óptima respetando horario laboral..."):
                    start_time = time.time()  # Tiempo de inicio
                    
                    # Obtener optimización de ruta
                    indices_ruta, distancia_total = vecino_mas_cercano_optimizado(
                        puntos_df, 
                        zona['centro_lat'], 
                        zona['centro_lng'],
                        max_puntos=max_puntos
                    )
                    
                    # Crear itinerario respetando horario laboral y colación
                    if indices_ruta:
                        itinerario, dias_totales = crear_itinerario_laboral(
                            puntos_df,
                            indices_ruta,
                            zona['centro_lat'],
                            zona['centro_lng'],
                            duracion_reunion=duracion_reunion,
                            tiempo_desplazamiento=tiempo_desplazamiento,
                            hora_inicio=hora_inicio,
                            hora_fin=hora_fin,
                            fecha_inicio=fecha_inicio,
                            dias_habiles=dias_laborales
                        )
                    else:
                        itinerario = pd.DataFrame()
                        dias_totales = 0
                    
                    # Calcular tiempo de procesamiento
                    tiempo_calculo = time.time() - start_time
                
                if len(itinerario) > 0:
                    # Mostrar métricas generales
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Puntos Totales", len(itinerario))
                    with col2:
                        st.metric("Días Necesarios", dias_totales)
                    with col3:
                        st.metric("Distancia Total", f"{distancia_total:.2f} km")
                    with col4:
                        st.metric("Tiempo de Cálculo", f"{tiempo_calculo:.2f} s")
                    
                    # Mostrar pestañas por día
                    dias_unicos = itinerario['Dia_Numero'].unique()
                    dias_unicos.sort()  # Asegurar orden ascendente
                    
                    if len(dias_unicos) > 0:
                        dias_tabs = st.tabs([f"Día {i}" for i in dias_unicos])
                        
                        mapas_por_dia = {}  # Almacenar mapas para evitar recálculos
                        
                        for i, tab_dia in enumerate(dias_tabs):
                            dia_num = dias_unicos[i]
                            with tab_dia:
                                # Filtrar itinerario para este día
                                itinerario_dia = itinerario[itinerario['Dia_Numero'] == dia_num]
                                
                                if not itinerario_dia.empty:
                                    # Información del día
                                    fecha_str = itinerario_dia['Fecha'].iloc[0]
                                    dia_semana = itinerario_dia['Dia_Semana'].iloc[0]
                                    
                                    # Crear mapa para este día
                                    if dia_num not in mapas_por_dia:
                                        mapa_dia = crear_mapa_ruta_dia(
                                            itinerario_dia, 
                                            zona['centro_lat'], 
                                            zona['centro_lng'],
                                            dia_num,
                                            fecha_str,
                                            dia_semana
                                        )
                                        mapas_por_dia[dia_num] = mapa_dia
                                    
                                    # Mostrar mapa
                                    st_folium(mapas_por_dia[dia_num], width=800, height=400)
                                    
                                    # Información básica del día
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Fecha", f"{fecha_str} ({dia_semana})")
                                    with col2:
                                        st.metric("Reuniones", len(itinerario_dia))
                                    with col3:
                                        # Calcular distancia del día
                                        distancia_dia = calcular_distancia_total_ruta(
                                            itinerario_dia,
                                            zona['centro_lat'], 
                                            zona['centro_lng']
                                        )
                                        st.metric("Distancia", f"{distancia_dia:.2f} km")
                                    
                                    # Tabla con itinerario del día
                                    st.subheader(f"Itinerario para Día {dia_num}")
                                    
                                    # Columnas a mostrar en el itinerario
                                    cols_itinerario = ['Orden_Dia', 'Nombre', 'ID registro', 'Direccion', 
                                                      'Hora_Llegada', 'Hora_Salida', 'lat', 'lng','Estado del lead', 'Etapa del ciclo de vida', 'Número de veces contactado', 'frecuencia_compra', 'cantidad_de_sucursales']
                                    
                                    # Añadir stage_name si está disponible
                                    #if 'stage_name' in itinerario_dia.columns:
                                    #    cols_itinerario.append('stage_name')
                                    
                                    # Verificar que todas las columnas existen
                                    cols_itinerario_final = [col for col in cols_itinerario if col in itinerario_dia.columns]
                                    
                                    st.dataframe(
                                        itinerario_dia[cols_itinerario_final],
                                        use_container_width=True,
                                        height=300
                                    )
                                    
                                    # Botón para descargar itinerario de este día
                                    csv_dia = itinerario_dia[cols_itinerario_final].to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label=f"📥 Descargar Itinerario Día {dia_num}",
                                        data=csv_dia,
                                        file_name=f"itinerario_zona_{index}_dia_{dia_num}.csv",
                                        mime="text/csv"
                                    )
                                    
                                    # Link a Google Maps para este día
                                    if len(itinerario_dia) > 0:
                                        # Limitar a 20 puntos por limitaciones de URL
                                        if len(itinerario_dia) > 20:
                                            puntos_mapa = itinerario_dia.head(20)
                                            st.info("Nota: Google Maps solo permite 20 waypoints. Mostrando los primeros 20 puntos de la ruta.")
                                        else:
                                            puntos_mapa = itinerario_dia
                                        
                                        # Construir URL
                                        origen = f"{zona['centro_lat']},{zona['centro_lng']}"
                                        destino = origen
                                        waypoints = "|".join([f"{row['lat']},{row['lng']}" for _, row in puntos_mapa.iterrows()])
                                        maps_url = f"https://www.google.com/maps/dir/?api=1&origin={origen}&destination={destino}&waypoints={waypoints}"
                                        
                                        st.markdown(f"[🗺️ Abrir ruta del día en Google Maps]({maps_url})")
                                else:
                                    st.info(f"No hay visitas programadas para el Día {dia_num}")
                    
                    # Mostrar resumen general
                    st.subheader("Resumen General del Itinerario")
                    
                    # Agrupar por día
                    resumen_dias = itinerario.groupby(['Dia_Numero', 'Fecha', 'Dia_Semana']).agg(
                        Visitas=('Orden_Dia', 'count')
                    ).reset_index()
                    
                    # Añadir distancia por día
                    distancias_dias = []
                    for _, row in resumen_dias.iterrows():
                        itinerario_dia = itinerario[itinerario['Dia_Numero'] == row['Dia_Numero']]
                        distancia_dia = calcular_distancia_total_ruta(
                            itinerario_dia,
                            zona['centro_lat'], 
                            zona['centro_lng']
                        )
                        distancias_dias.append(distancia_dia)
                    
                    resumen_dias['Distancia_km'] = distancias_dias
                    
                    # Mostrar tabla de resumen
                    st.dataframe(
                        resumen_dias.rename(columns={
                            'Dia_Numero': 'Día', 
                            'Dia_Semana': 'Día Semana',
                            'Distancia_km': 'Distancia (km)'
                        }),
                        use_container_width=True
                    )
                    
                    # Botones de descarga
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Descargar itinerario completo
                        csv_completo = itinerario.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Descargar Itinerario Completo (CSV)",
                            data=csv_completo,
                            file_name=f"itinerario_completo_zona_{index}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Crear texto para informe
                        texto_informe = f"# Informe de Visitas para Zona {index}: {zona['nombre_zona']}\n\n"
                        texto_informe += f"**Puntos totales a visitar:** {len(itinerario)}\n"
                        texto_informe += f"**Días necesarios:** {dias_totales}\n"
                        texto_informe += f"**Distancia total:** {distancia_total:.2f} km\n\n"
                        
                        texto_informe += "## Resumen por Día\n\n"
                        texto_informe += "| Día | Fecha | Día Semana | Visitas | Distancia (km) |\n"
                        texto_informe += "|-----|-------|------------|---------|---------------|\n"
                        
                        for _, row in resumen_dias.iterrows():
                            texto_informe += f"| {row['Dia_Numero']} | {row['Fecha']} | {row['Dia_Semana']} | {row['Visitas']} | {row['Distancia_km']:.2f} |\n"
                        
                        texto_informe += "\n\n## Itinerario Detallado\n\n"
                        
                        for dia in dias_unicos:
                            itinerario_dia = itinerario[itinerario['Dia_Numero'] == dia]
                            if not itinerario_dia.empty:
                                fecha_str = itinerario_dia['Fecha'].iloc[0]
                                dia_semana = itinerario_dia['Dia_Semana'].iloc[0]
                                
                                texto_informe += f"### Día {dia}: {fecha_str} ({dia_semana})\n\n"
                                texto_informe += "| # | Nombre | Dirección | Llegada | Salida |\n"
                                texto_informe += "|---|--------|-----------|---------|--------|\n"
                                
                                for _, row in itinerario_dia.iterrows():
                                    nombre = row.get('Nombre', f"Punto {row['Orden_Dia']}")
                                    direccion = row.get('Direccion', 'Sin dirección')
                                    llegada = row.get('Hora_Llegada', '')
                                    salida = row.get('Hora_Salida', '')
                                    
                                    texto_informe += f"| {row['Orden_Dia']} | {nombre} | {direccion} | {llegada} | {salida} |\n"
                                
                                texto_informe += "\n"
                        
                        # Botón para descargar informe en texto
                        texto_data = texto_informe.encode('utf-8')
                        st.download_button(
                            label=f"📥 Descargar Informe Completo (Texto)",
                            data=texto_data,
                            file_name=f"informe_zona_{index}.md",
                            mime="text/markdown"
                        )
                
                else:
                    st.warning("No se pudo generar una ruta óptima para esta zona.")
            else:
                st.warning("No hay puntos disponibles para crear rutas en esta zona después de filtrar clientes activos.")
        
    except Exception as e:
        st.error(f"Error al mostrar detalle de zona con ruta: {str(e)}")
        st.code(traceback.format_exc())

# Función principal
def main():
    try:
        # Configurar título
        st.title("🔥 Optimizador de Rutas para Prospección en CDMX")
        st.markdown("Genera rutas óptimas para visitar todos los puntos en cada zona, respetando horario laboral y días hábiles")
        
        # Sidebar para configuración
        with st.sidebar:
            st.header("Configuración de Zonas")
            radio_max = st.slider("Radio máximo (km)", 1.0, 2.5, 2.0, 0.1)
            min_puntos = st.slider("Mínimo de puntos por zona", 3, 15, 5, 1)
            mostrar_puntos = st.checkbox("Mostrar puntos en mapa general", True)
            
            st.header("Parámetros de Horario Laboral")
            hora_inicio = st.text_input("Hora inicio", "09:00")
            hora_fin = st.text_input("Hora fin", "18:00")
            st.info("Tiempo de colación (almuerzo): 13:00 - 14:00")
            
            st.header("Parámetros de Visitas")
            duracion_visita = st.slider("Duración de cada visita (min)", 15, 60, 30, 5)
            tiempo_desplaz = st.slider("Tiempo de desplazamiento (min)", 5, 30, 10, 5)
            
            # Días laborales
            st.write("Días laborales predeterminados:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("✓ Lunes a Viernes")
            with col2:
                st.write("✗ Sábado y Domingo")
            
            # Guardar configuración para usar en funciones
            config = {
                'hora_inicio': hora_inicio,
                'hora_fin': hora_fin,
                'duracion_visita': duracion_visita,
                'tiempo_desplaz': tiempo_desplaz,
                'dias_laborales': [0, 1, 2, 3, 4]  # Lun-Vie por defecto
            }
            st.session_state['config_laboral'] = config
        
        # Cargar datos
        with st.spinner('Cargando datos...'):
            df = cargar_datos()
            
            if df.empty:
                st.error("No se pudieron cargar los datos.")
                return
            
            # Asignar ponderaciones
            df = asignar_ponderaciones(df)
        
        # Mostrar estadísticas básicas
        total_puntos = len(df)
        st.metric("Total Puntos", f"{total_puntos:,}")
        
        # Identificar hot-spots
        with st.spinner("Identificando zonas óptimas..."):
            resultados_df, cluster_stats = crear_hotspots_downtown(
                df,
                radio_max_km=radio_max,
                min_puntos=min_puntos,
                max_zonas=5
            )
        
        # Mostrar resultados
        if resultados_df is not None and not resultados_df.empty:
            # Mapa principal
            st.subheader("Mapa de Zonas para Prospección")
            
            fig = visualizar_hotspots(cluster_stats, mostrar_puntos=mostrar_puntos)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de resultados
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(resultados_df, use_container_width=True, height=200)
            
            with col2:
                st.info("Estas zonas representan las mejores áreas para prospección comercial. Cada zona está planificada para un vendedor, respetando horario laboral y días hábiles.")
            
            # Pestañas para zonas
            tabs = st.tabs([f"Zona {i+1}: {cluster_stats[i]['nombre_zona']}" for i in range(len(cluster_stats))])
            
            for i, tab in enumerate(tabs):
                with tab:
                    mostrar_detalle_zona_con_ruta_laboral(cluster_stats[i], i+1)
        else:
            st.warning("No se pudieron identificar zonas con los parámetros establecidos. Intente ajustar los parámetros.")
    
    except Exception as e:
        st.error(f"Error en la aplicación: {str(e)}")
        st.error(traceback.format_exc())

# Información sobre la aplicación
st.sidebar.markdown("""
### Información de la Aplicación
- Esta aplicación optimiza rutas para visitar puntos en zonas específicas
- Respeta horario laboral (9:00-18:00 por defecto)
- Incluye tiempo de colación/almuerzo (13:00-14:00)
- Distribuye las visitas solo en días laborales (Lun-Vie por defecto)
- Cada zona se asigna a un vendedor diferente
- Las rutas comienzan y terminan en el centro de cada zona
- El itinerario incluye estimaciones de tiempo de visita
""")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()