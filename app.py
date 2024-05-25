import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.sparse import csr_matrix
from minisom import MiniSom
import base64
import folium
from streamlit_folium import folium_static
from folium import plugins

# streamlit run app.py


def cargar_datos():
    data_original = pd.read_excel('completo_codificado.xlsx')
    return data_original

def cargar_datos_total():
    data_total = pd.read_excel('completo_total.xlsx')
    return data_total


def preprocesar_datos(data_original, features_intrinsecas):
    data_intrinsecas = data_original[features_intrinsecas]

    numeric_features = data_intrinsecas.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = list(set(data_intrinsecas.columns) - set(numeric_features))

    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    data_preprocessed = preprocessor.fit_transform(data_intrinsecas)
    return data_preprocessed, preprocessor


def entrenar_som(data_preprocessed):
    if isinstance(data_preprocessed, csr_matrix):
        data_preprocessed_dense = data_preprocessed.toarray()
    else:
        data_preprocessed_dense = data_preprocessed

    params = {'x': 15, 'y': 15, 'sigma': 0.7, 'learning_rate': 0.5}
    som = MiniSom(x=params['x'], y=params['y'], input_len=data_preprocessed_dense.shape[1], sigma=params['sigma'],
                  learning_rate=params['learning_rate'], random_seed=42)
    som.train_random(data_preprocessed_dense, 20000)
    return som, data_preprocessed_dense


def create_map(df):
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=12, tiles=None)

    # Otro estilo: cartodbpositron
    folium.TileLayer('openstreetmap').add_to(folium_map)

    # Agregar un mini mapa para facilitar la navegación
    minimap = plugins.MiniMap()
    folium_map.add_child(minimap)

    for _, row in df.iterrows():
        popup_html = f"<b>Dirección:</b> {row['address']}<br>"
        popup_html += f"<b>Price:</b> {row['price']}<br>"
        popup_html += f"<b>Property Type:</b> {row['propertyType']}<br>"
        popup_html += f"<b>Size:</b> {row['size']}<br>"
        popup_html += f"<b>Rooms:</b> {row['rooms']}<br>"
        popup_html += f"<b>Bathrooms:</b> {row['bathrooms']}"

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(folium_map)

    return folium_map



def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def estrellas_valoracion(nota):
    estrellas_totales = 5
    estrella_llena = '⭐️'  # Estrella llena
    estrella_vacia = '✩ '  # Estrella vacía que coincida en estilo con la estrella llena
    if nota is None:
        estrellas = estrella_vacia * estrellas_totales  # Indicador visual de no disponible
    else:
        # Se calcula la cantidad de estrellas llenas basado en la nota y la cantidad total de estrellas
        estrellas_llenas = round((nota + 19) / 35 * estrellas_totales)
        estrellas = estrella_llena * estrellas_llenas + estrella_vacia * (estrellas_totales - estrellas_llenas)
    return estrellas

def mostrar_recomendaciones_y_mapa(df, titulo, puntos_de_interes_seleccionados):
    df['nota_media'] = df['Peso'].apply(lambda x: round(x * 10 / 26 + 5, 1) if x is not None else None)
    df_filtrado = df[df['nota_media'] > 5].head(10)

    if not df_filtrado.empty:
        st.markdown(f"## {titulo}")
        st.markdown("---")  # Agregar un separador

        # Inicializa el contador de viviendas
        contador_viviendas = 1

        for index, row in df_filtrado.iterrows():
            peso = row.get('Peso')
            estrellas = estrellas_valoracion(peso)
            nota_media = round(peso * 10 / 26 + 5, 1) if peso is not None else 'N/A'

            # Orden
            html_numero_orden = f"<div style='margin: 10px; padding: 10px;'><h3>{contador_viviendas}.</h3></div>"

            # Creación del HTML para mostrar detalles de la vivienda
            html_detalles_vivienda = f"""
                    <div style="margin: 10px; padding: 10px; border-radius: 5px">
                        <h4>Características de la vivienda</h4>
                        <ul>
                            <li><b>Dirección:</b> {row['address']}</li>
                            <li><b>Precio:</b> {row['price']} €</li>
                            <li><b>Tipo de Propiedad:</b> {row['propertyType']}</li>
                            <li><b>Tamaño:</b> {row['size']} m²</li>
                            <li><b>Habitaciones:</b> {row['rooms']}</li>
                            <li><b>Baños:</b> {row['bathrooms']}</li>
                            <li><b>Piso:</b> {row.get('floor', 'N/A')}</li>
                        </ul>
                    </div>
                    <div style="margin: 10px; padding: 10px;">
                        <h4>Valoraciones de los clientes</h4>
                        <ul>
                            <li>Puntuación: {estrellas}</li>
                            <li>Nota media: {nota_media}</li>
                            <li>Reseñas: {row.get('Valoracion_Sintetica', 'N/A')}</li>
                        </ul>
                    </div>
            """

            # Sección de puntos de interés
            if puntos_de_interes_seleccionados:
                html_puntos_interes = '<div style="margin: 10px; padding: 10px;"><h4>Puntos de interés cercanos</h4><ul>'
                for punto_de_interes in puntos_de_interes_seleccionados:
                    punto_nombre = punto_de_interes.replace('closest_', '')
                    distancia_columna = f'distance_to_{punto_nombre}_km'
                    if punto_de_interes in row and distancia_columna in row:
                        nombre_punto_interes = row[punto_de_interes]
                        distancia = int(row[distancia_columna])
                        html_puntos_interes += f"<li>{nombre_punto_interes} ({distancia} km)</li>"
                html_puntos_interes += "</ul></div>"
            else:
                html_puntos_interes = ""  # Deja esta sección vacía si no hay puntos de interés seleccionados

            # Características adicionales presentes en la vivienda
            html_caracteristicas_adicionales = '<div style="margin: 10px; padding: 10px;"><h4>Características adicionales</h4><ul>'
            caracteristicas_adicionales = ['exterior', 'hasLift', 'piscina', 'aire acondicionado', 'trastero',
                                           'chimenea', 'sin amueblar', 'electrodomésticos', 'portero', 'mascotas']
            for caracteristica in caracteristicas_adicionales:
                if row.get(caracteristica, 0) == 1:
                    html_caracteristicas_adicionales += f"<li>{caracteristica.replace('_', ' ').capitalize()}</li>"
            html_caracteristicas_adicionales += "</ul></div>"

            # Combina todas las secciones en un solo HTML
            html_total = html_numero_orden + html_detalles_vivienda + html_puntos_interes + html_caracteristicas_adicionales
            st.markdown(html_total, unsafe_allow_html=True)

            # Incrementa el contador de viviendas para la próxima iteración
            contador_viviendas += 1

            st.markdown("---")  # Agregar un separador

        # Mostrar el mapa solo una vez al final, si es necesario
        mapa = create_map(df_filtrado)
        folium_static(mapa)

    else:
        html_valoraciones = ('<div style="margin: 10px; padding: 10px;"><h4>Las valoraciones de las viviendas '
        
                             'disponibles no son lo suficientemente altas para dar una recomendación</h4>')

        st.markdown(html_valoraciones, unsafe_allow_html=True)


def aplicacion_streamlit():
    # Establece la imagen de fondo
    set_png_as_page_bg('fondo.jpg')
    st.title("Recomendación de viviendas: PriceIt")

    # Carga de datos y preparación
    data_original = cargar_datos()
    provinces = sorted(data_original['province'].unique().tolist())

    # Configurar pestañas
    tab1, tab2 = st.tabs(["Seleccionar Provincia", "Detalles de la Vivienda"])

    with tab1:
        # Selector de provincia
        selected_province = st.selectbox('Selecciona tu provincia', provinces)
        # Guardar la selección en el estado de la sesión para usarla en la siguiente pestaña
        if st.button('Confirmar Provincia'):
            st.session_state['province'] = selected_province
            st.success(
                f"Provincia {selected_province} seleccionada. Por favor, ve a la pestaña 'Detalles de la Vivienda'.")

    with tab2:

        if 'province' in st.session_state:
            st.write(f"Provincia seleccionada: {st.session_state['province']}")

            # Filtrar data_original para la provincia seleccionada
            filtered_data = data_original[data_original['province'] == st.session_state['province']]
            property_types_english = filtered_data['propertyType'].unique().tolist()

            # Diccionario para mapear los tipos de propiedad del inglés al español
            traducciones_propiedades = {
                'chalet': 'Chalet',
                'flat': 'Piso',
                'studio': 'Estudio',
                'countryHouse': 'Casa Rural',
                'penthouse': 'Ático',
                'duplex': 'Dúplex'
            }

            # Traducir los tipos de propiedad al español para la visualización
            property_types_spanish = [traducciones_propiedades[tipo] for tipo in property_types_english]

            # Entradas del usuario
            price = st.number_input('Precio', min_value=0, value=0, format='%d')
            property_type_spanish_selected = st.selectbox('Tipo de Propiedad', property_types_spanish)
            rooms = st.number_input('Número de Habitaciones', min_value=0, value=0, format='%d')
            bathrooms = st.number_input('Número de Baños', min_value=0, value=0, format='%d')

            # Convertir la selección en español de vuelta al valor original en inglés
            property_type = None
            for english, spanish in traducciones_propiedades.items():
                if property_type_spanish_selected == spanish:
                    property_type = english
                    break

            # Asegurar que la opción de piso solo está disponible si la propiedad es "flat"
            if property_type == 'flat' or property_type == 'studio':
                floor = st.number_input('Piso', min_value=0, value=0, format='%d')
            else:
                floor = 0  # Asumimos 0 como valor por defecto para propiedades que no son "flat"

            need_hospital = st.checkbox('¿Necesitas estar cerca de un hospital?')
            if need_hospital:
                closest_hospitals_sorted = sorted(filtered_data['closest_hospital'].unique())
                closest_hospital = st.selectbox('Hospital más cercano', closest_hospitals_sorted)
            else:
                closest_hospital = None

            need_estacion = st.checkbox('¿Necesitas estar cerca de una estación?')
            if need_estacion:
                closest_estacion_sorted = sorted(filtered_data['closest_estacion'].unique())
                closest_estacion = st.selectbox('Estación más cercana', closest_estacion_sorted)
            else:
                closest_estacion = None

            need_aeropuerto = st.checkbox('¿Necesitas estar cerca de un aeropuerto?')
            if need_aeropuerto:
                closest_aeropuerto_sorted = sorted(filtered_data['closest_aeropuerto'].unique())
                closest_aeropuerto = st.selectbox('Aeropuerto más cercano', closest_aeropuerto_sorted)
            else:
                closest_aeropuerto = None

            need_playa = st.checkbox('¿Necesitas estar cerca de una playa?')
            if need_playa:
                closest_playa_sorted = sorted(filtered_data['closest_playa'].unique())
                closest_playa = st.selectbox('Playa más cercana', closest_playa_sorted)
            else:
                closest_playa = None

            need_universidad = st.checkbox('¿Necesitas estar cerca de una universidad?')
            if need_universidad:
                closest_universidad_sorted = sorted(filtered_data['closest_universidad'].unique())
                closest_universidad = st.selectbox('Universidad más cercana', closest_universidad_sorted)
            else:
                closest_universidad = None

            caracteristicas_adicionales = {
                'exterior': st.checkbox('¿Quieres que tenga terraza?'),
                'hasLift': st.checkbox('¿Necesitas ascensor?'),
                'piscina': st.checkbox('¿Buscas una piscina?'),
                'aire acondicionado': st.checkbox('¿Sería necesario tener aire acondicionado?'),
                'trastero': st.checkbox('¿Es necesario un trastero?'),
                'chimenea': st.checkbox('¿Quieres chimenea?'),
                'sin amueblar': st.checkbox('¿Lo prefieres sin amueblar?'),
                'electrodomésticos': st.checkbox('¿Debe incluir electrodomésticos?'),
                'portero': st.checkbox('¿Te gustaría que tuviera servicio de portero?'),
                'mascotas': st.checkbox('¿Debe ser apto para mascotas?')
            }

            features_intrinsecas = ['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province', 'floor']
            muestra = {'price': price, 'propertyType': property_type, 'size': 100, 'rooms': rooms,
                       'bathrooms': bathrooms, 'province': selected_province, 'floor': floor,
                       'closest_hospital': closest_hospital, 'closest_estacion': closest_estacion,
                       'closest_aeropuerto': closest_aeropuerto, 'closest_playa': closest_playa,
                       'closest_universidad': closest_universidad}
            muestra.update(caracteristicas_adicionales)

            if st.button('Obtener Recomendaciones'):
                data_original = cargar_datos()
                data_total = cargar_datos_total()
                data_preprocessed, preprocessor = preprocesar_datos(data_original, features_intrinsecas)
                som, data_preprocessed_dense = entrenar_som(data_preprocessed)

                muestra_df = pd.DataFrame([muestra])[features_intrinsecas]
                muestra_preprocessed = preprocessor.transform(muestra_df)

                if isinstance(muestra_preprocessed, csr_matrix):
                    muestra_preprocessed_dense = muestra_preprocessed.toarray()
                else:
                    muestra_preprocessed_dense = muestra_preprocessed

                win_map = som.win_map(data_preprocessed_dense)
                winner = som.winner(muestra_preprocessed_dense[0])
                similar_houses = win_map[winner]

                similar_indices = [np.argwhere(np.all(data_preprocessed_dense == house, axis=1))[0][0] for house in
                                   similar_houses]
                recomendaciones_df_original = data_original.iloc[similar_indices].copy()

                # Copia para mantener el dataframe original antes de aplicar filtros
                recomendaciones_df = recomendaciones_df_original.copy()

                # Crear DataFrames para cada filtro de punto de interés
                recomendaciones_hospital = recomendaciones_df_original[recomendaciones_df_original[
                                                                           'closest_hospital'] == closest_hospital] if need_hospital else pd.DataFrame()
                recomendaciones_estacion = recomendaciones_df_original[recomendaciones_df_original[
                                                                           'closest_estacion'] == closest_estacion] if need_estacion else pd.DataFrame()
                recomendaciones_aeropuerto = recomendaciones_df_original[recomendaciones_df_original[
                                                                             'closest_aeropuerto'] == closest_aeropuerto] if need_aeropuerto else pd.DataFrame()
                recomendaciones_playa = recomendaciones_df_original[
                    recomendaciones_df_original['closest_playa'] == closest_playa] if need_playa else pd.DataFrame()

                recomendaciones_universidad = recomendaciones_df_original[recomendaciones_df_original[
                                                                              'closest_universidad'] == closest_universidad] if need_universidad else pd.DataFrame()

                # Aplicar filtros sobre el dataframe original para el dataframe final
                puntos_de_interes_seleccionados = []
                if need_hospital:
                    recomendaciones_df = recomendaciones_df[recomendaciones_df['closest_hospital'] == closest_hospital]
                    puntos_de_interes_seleccionados.append('closest_hospital')
                if need_estacion:
                    recomendaciones_df = recomendaciones_df[recomendaciones_df['closest_estacion'] == closest_estacion]
                    puntos_de_interes_seleccionados.append('closest_estacion')
                if need_aeropuerto:
                    recomendaciones_df = recomendaciones_df[
                        recomendaciones_df['closest_aeropuerto'] == closest_aeropuerto]
                    puntos_de_interes_seleccionados.append('closest_aeropuerto')
                if need_playa:
                    recomendaciones_df = recomendaciones_df[recomendaciones_df['closest_playa'] == closest_playa]
                    puntos_de_interes_seleccionados.append('closest_playa')
                if need_universidad:
                    recomendaciones_df = recomendaciones_df[
                        recomendaciones_df['closest_universidad'] == closest_universidad]
                    puntos_de_interes_seleccionados.append('closest_universidad')

                for caracteristica, necesidad in caracteristicas_adicionales.items():
                    if necesidad:  # Si el usuario necesita esta característica
                        recomendaciones_df = recomendaciones_df[recomendaciones_df[caracteristica] == 1]

                if recomendaciones_df.empty:
                    # Mostrar recomendaciones alternativas si el dataframe final está vacío
                    st.write(
                        "#### No se encontraron recomendaciones que cumplan con todos los requisitos seleccionados. "
                        "Considere estas alternativas:")
                    st.markdown("---")  # Agregar un separador

                    # Inicialización de variables de coordenadas con DataFrames vacíos
                    coordenadas_hospitales = pd.DataFrame()
                    coordenadas_estaciones = pd.DataFrame()
                    coordenadas_playa = pd.DataFrame()
                    coordenadas_aeropuerto = pd.DataFrame()
                    coordenadas_universidad = pd.DataFrame()

                    coordenadas_mias = pd.merge(
                        recomendaciones_df_original,
                        data_total[['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province', 'latitude',
                                    'longitude', 'address', 'Valoracion_Sintetica']],
                        on=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'],
                        how='left'
                    )

                    coordenadas_mias.drop_duplicates(
                        subset=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'], inplace=True)


                    #for caracteristica, necesidad in caracteristicas_adicionales.items():
                    #    if necesidad:  # Si el usuario necesita esta característica
                    #        coordenadas_mias = coordenadas_mias[coordenadas_mias[caracteristica] == 1]


                    mostrar_recomendaciones_y_mapa(coordenadas_mias, "Nuestras recomendaciones:",
                                                   puntos_de_interes_seleccionados)


                    if not recomendaciones_hospital.empty:
                        st.markdown("---")  # Agregar un separador

                        coordenadas_hospitales = pd.merge(
                            recomendaciones_hospital,
                            data_total[['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province', 'latitude',
                                        'longitude', 'address', 'Valoracion_Sintetica']],
                            on=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'],
                            how='left'
                        )

                        coordenadas_hospitales.drop_duplicates(
                            subset=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'], inplace=True)

                        # Ordenar por la columna 'distance_to_estacion_km' de menor a mayor distancia
                        coordenadas_hospitales.sort_values(by='distance_to_hospital_km', inplace=True)

                        # Mostrar recomendaciones
                        mostrar_recomendaciones_y_mapa(coordenadas_hospitales, "Recomendaciones cerca del "
                                                                               "hospital seleccionado:",
                                                       puntos_de_interes_seleccionados)

                    if not recomendaciones_estacion.empty:
                        st.markdown("---")  # Agregar un separador

                        coordenadas_estaciones = pd.merge(
                            recomendaciones_estacion,
                            data_total[['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province', 'latitude',
                                        'longitude', 'address', 'Valoracion_Sintetica']],
                            on=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'],
                            how='left'
                        )

                        coordenadas_estaciones.drop_duplicates(
                            subset=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'], inplace=True)

                        # Ordenar por la columna 'distance_to_estacion_km' de menor a mayor distancia
                        coordenadas_estaciones.sort_values(by='distance_to_estacion_km', inplace=True)

                        # Mostrar recomendaciones
                        mostrar_recomendaciones_y_mapa(coordenadas_estaciones, "Recomendaciones cerca de "
                                                                               "la estación seleccionada:",
                                                       puntos_de_interes_seleccionados)

                    if not recomendaciones_playa.empty:
                        st.markdown("---")  # Agregar un separador

                        coordenadas_playa = pd.merge(
                            recomendaciones_playa,
                            data_total[['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province', 'latitude',
                                        'longitude', 'address', 'Valoracion_Sintetica']],
                            on=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'],
                            how='left'
                        )

                        coordenadas_playa.drop_duplicates(
                            subset=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'], inplace=True)

                        # Ordenar por la columna 'distance_to_estacion_km' de menor a mayor distancia
                        coordenadas_playa.sort_values(by='distance_to_playa_km', inplace=True)

                        # Mostrar recomendaciones
                        mostrar_recomendaciones_y_mapa(coordenadas_playa, "Recomendaciones cerca de"
                                                                          " la playa seleccionada:",
                                                       puntos_de_interes_seleccionados)

                    if not recomendaciones_aeropuerto.empty:
                        st.markdown("---")  # Agregar un separador

                        coordenadas_aeropuerto = pd.merge(
                            recomendaciones_aeropuerto,
                            data_total[['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province', 'latitude',
                                        'longitude', 'address', 'Valoracion_Sintetica']],
                            on=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'],
                            how='left'
                        )

                        coordenadas_aeropuerto.drop_duplicates(
                            subset=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'], inplace=True)

                        # Ordenar por la columna 'distance_to_estacion_km' de menor a mayor distancia
                        coordenadas_aeropuerto.sort_values(by='distance_to_aeropuerto_km', inplace=True)

                        # Mostrar recomendaciones
                        mostrar_recomendaciones_y_mapa(coordenadas_aeropuerto, "Recomendaciones cerca del"
                                                                               " aeropuerto seleccionado:",
                                                       puntos_de_interes_seleccionados)

                    if not recomendaciones_universidad.empty:
                        st.markdown("---")  # Agregar un separador

                        coordenadas_universidad = pd.merge(
                            recomendaciones_universidad,
                            data_total[['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province', 'latitude',
                                        'longitude', 'address', 'Valoracion_Sintetica']],
                            on=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'],
                            how='left'
                        )

                        coordenadas_universidad.drop_duplicates(
                            subset=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'], inplace=True)

                        # Ordenar por la columna 'distance_to_estacion_km' de menor a mayor distancia
                        coordenadas_universidad.sort_values(by='distance_to_universidad_km', inplace=True)

                        # Mostrar recomendaciones
                        mostrar_recomendaciones_y_mapa(coordenadas_universidad, "Recomendaciones cerca de "
                                                                                "la universidad seleccionada:",
                                                       puntos_de_interes_seleccionados)

                    # RECOMENDACIONES MÁS REPETIDAS
                    # 1. Identificar DataFrames no vacíos y contarlos
                    dataframes_no_vacios = [df for df in
                                            [coordenadas_mias, coordenadas_hospitales, coordenadas_estaciones,
                                             coordenadas_playa, coordenadas_aeropuerto, coordenadas_universidad] if
                                            not df.empty]

                    # 2. Concatenar los DataFrames no vacíos
                    df_concatenado = pd.concat(dataframes_no_vacios).copy()

                    # 3. Contar la frecuencia de aparición de cada vivienda
                    frecuencia_viviendas = df_concatenado['address'].value_counts()

                    # 4. Filtrar las viviendas que alcanzan o superan la frecuencia mínima
                    viviendas_frecuentes = frecuencia_viviendas[
                        frecuencia_viviendas >= 2].index.tolist()

                    # 5. Filtrar el DataFrame concatenado para incluir solo las viviendas frecuentes
                    recomendaciones_finales = df_concatenado[
                        df_concatenado['address'].isin(viviendas_frecuentes)].copy()

                    # Paso adicional para ordenar por frecuencia de aparición
                    # Crear una columna 'frecuencia' en recomendaciones_finales basada en la frecuencia de 'address'
                    recomendaciones_finales['frecuencia'] = recomendaciones_finales['address'].map(frecuencia_viviendas)

                    # Ordenar recomendaciones_finales por 'frecuencia', de mayor a menor
                    recomendaciones_finales = recomendaciones_finales.sort_values(by='frecuencia', ascending=False)

                    # Eliminar la columna 'frecuencia' si no deseas mostrarla y otros duplicados
                    recomendaciones_finales = recomendaciones_finales.drop(columns=['frecuencia']).drop_duplicates(
                        subset=['address'])

                    # Mostrar las recomendaciones finales, ahora ordenadas por frecuencia de aparición
                    st.markdown("---")  # Agregar un separador
                    mostrar_recomendaciones_y_mapa(recomendaciones_finales,
                                                   "Recomendaciones finales ajustadas a sus preferencias:",
                                                   puntos_de_interes_seleccionados)


                else:
                    recomendaciones_con_coordenadas = pd.merge(
                        recomendaciones_df,
                        data_total[['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province', 'latitude',
                                    'longitude', 'address', 'Valoracion_Sintetica']],
                        on=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'],
                        how='left'
                    )

                    recomendaciones_con_coordenadas.drop_duplicates(
                        subset=['price', 'propertyType', 'size', 'rooms', 'bathrooms', 'province'], inplace=True)

                    mostrar_recomendaciones_y_mapa(recomendaciones_con_coordenadas, "Recomendaciones basadas en"
                                                                                    " tus preferencias y puntos de "
                                                                                    "interés seleccionados:",
                                                   puntos_de_interes_seleccionados)
        else:
                st.warning("Por favor, selecciona primero una provincia en la pestaña 'Seleccionar Provincia'.")

# Llama a la función para ejecutar la aplicación
if __name__ == '__main__':
    aplicacion_streamlit()

