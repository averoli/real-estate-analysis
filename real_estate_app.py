# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Set page configuration
st.set_page_config(
    page_title="Real Estate Analysis",
    page_icon="📊",
    layout="wide"
)

# Configure matplotlib with a simple style
plt.rcParams.update({
    'figure.figsize': [10, 6],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 12,
    'axes.titlesize': 14
})

# === Column Templates ===
STANDARD_COLUMNS = {
    "Комплекс": ["Комплекс", "Project", "Name"],
    "Район": ["Район", "District", "Area"],
    "Площадь": ["Площадь", "Площадь (м²)", "Area (sqm)"],
    "Спален": ["Спален", "Bedrooms"],
    "Этажей": ["Этажей", "Floors"],
    "Цена_฿": ["Price, ฿", "Цена", "Price (THB)"]
}

st.title("📥 Объединение и анализ Excel-файлов недвижимости")

def read_file_safely(file, is_csv=True, header_row=0):
    try:
        if is_csv:
            # Try different encodings
            encodings = ['utf-8', 'cp1251', 'latin1']
            for encoding in encodings:
                try:
                    return pd.read_csv(file, header=header_row, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not read the file with any supported encoding")
        else:
            return pd.read_excel(file, header=header_row)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

uploaded_files = st.file_uploader(
    "Загрузите один или несколько Excel-файлов", 
    type=["xlsx", "xls", "csv"], 
    accept_multiple_files=True
)

if uploaded_files:
    all_data = []
    
    for file in uploaded_files:
        st.subheader(f"📄 Файл: {file.name}")
        
        try:
            is_csv = file.name.endswith('.csv')
            
            # Preview the file
            if is_csv:
                df_preview = read_file_safely(file, is_csv=True, header_row=None)
            else:
                df_preview = read_file_safely(file, is_csv=False, header_row=None)
            
            if df_preview is None:
                st.warning(f"Не удалось прочитать файл {file.name}")
                continue
                
            st.write("🧾 Предпросмотр:", df_preview.head())
            
            # Get header row from user
            header_row = st.number_input(
                f"🧩 В какой строке заголовки в {file.name}?",
                min_value=0,
                max_value=10,
                value=0 if is_csv else 1,
                key=f"header_{file.name}"
            )
            
            # Read the full file with the specified header
            df = read_file_safely(file, is_csv=is_csv, header_row=header_row)
            
            if df is None or df.empty:
                st.warning(f"Файл {file.name} пуст или не содержит данных")
                continue
            
            st.write("🔎 Найденные колонки:", list(df.columns))

            # Column mapping
            column_map = {}
            for std_col, aliases in STANDARD_COLUMNS.items():
                match = next((alias for alias in aliases if alias in df.columns), None)
                column_map[std_col] = st.selectbox(
                    f"🧩 Сопоставьте колонку для '{std_col}'",
                    options=[""] + list(df.columns),
                    index=(list(df.columns).index(match) + 1) if match else 0,
                    key=f"{file.name}_{std_col}"
                )

            # Create mapped dataframe
            mapped_df = pd.DataFrame()
            for std_col, selected_col in column_map.items():
                if selected_col:
                    mapped_df[std_col] = df[selected_col]
                else:
                    mapped_df[std_col] = None

            all_data.append(mapped_df)
            
        except Exception as e:
            st.error(f"Ошибка при обработке файла {file.name}: {str(e)}")
            continue

    if not all_data:
        st.warning("Не удалось обработать ни один файл")
        st.stop()

    try:
        # Combine all files
        result_df = pd.concat(all_data, ignore_index=True)
        if result_df.empty:
            st.warning("После объединения файлов данные отсутствуют")
            st.stop()
            
        st.success("✅ Файлы успешно объединены!")
        st.dataframe(result_df)

        # === Price calculations ===
        try:
            result_df['Цена_฿'] = pd.to_numeric(
                result_df['Цена_฿'].astype(str).str.replace(' ', '').str.replace(',', ''),
                errors='coerce'
            )
            result_df['Площадь'] = pd.to_numeric(result_df['Площадь'], errors='coerce')
            result_df['Цена_за_м2'] = result_df['Цена_฿'] / result_df['Площадь']
        except Exception as e:
            st.warning(f"Ошибка при расчете цены за м²: {str(e)}")

        # === 💡 Анализ выгодных объектов ===
        st.header("💡 Выгодные объекты (цена за м² < 80% от средней по району)")
        if 'Район' in result_df.columns and 'Цена_за_м2' in result_df.columns:
            avg_by_area = result_df.groupby('Район')['Цена_за_м2'].mean()
            result_df['Отклонение_от_средней'] = result_df.apply(
                lambda row: row['Цена_за_м2'] / avg_by_area[row['Район']]
                if pd.notnull(row['Район']) and row['Район'] in avg_by_area else None,
                axis=1
            )
            profitable_df = result_df[result_df['Отклонение_от_средней'] < 0.8]
            st.dataframe(profitable_df)

        # === 📈 График средней цены по районам ===
        st.header("📈 Средняя цена за м² по районам")
        if 'Район' in result_df.columns and 'Цена_за_м2' in result_df.columns:
            avg_price_chart = result_df.groupby('Район')['Цена_за_м2'].mean().sort_values()
            fig, ax = plt.subplots(figsize=(10, 5))
            avg_price_chart.plot(kind='barh', ax=ax)
            ax.set_xlabel("Цена за м² (฿)")
            ax.set_title("Средняя цена за м² по районам")
            st.pyplot(fig)

        # === 🤖 AI-прогноз цены ===
        st.header("🤖 Прогноз цены по параметрам")
        model_data = result_df.dropna(subset=['Площадь', 'Спален', 'Этажей', 'Район', 'Цена_฿'])

        if not model_data.empty:
            X = model_data[['Площадь', 'Спален', 'Этажей', 'Район']]
            y = model_data['Цена_฿']

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Район'])
            ], remainder='passthrough')

            model = Pipeline([
                ('prep', preprocessor),
                ('rf', RandomForestRegressor(random_state=42))
            ])

            model.fit(X, y)

            col1, col2 = st.columns(2)
            with col1:
                input_area = st.number_input("Площадь (м²)", min_value=20, max_value=1000, value=150)
                input_bedrooms = st.number_input("Спален", min_value=1, max_value=10, value=3)
            with col2:
                input_floors = st.number_input("Этажей", min_value=1, max_value=5, value=1)
                input_district = st.selectbox("Район", sorted(model_data['Район'].dropna().unique()))

            input_df = pd.DataFrame([{
                'Площадь': input_area,
                'Спален': input_bedrooms,
                'Этажей': input_floors,
                'Район': input_district
            }])

            predicted_price = model.predict(input_df)[0]
            st.subheader(f"💰 Прогнозируемая цена: **฿ {int(predicted_price):,}**")
        else:
            st.warning("Недостаточно данных для прогнозирования.")

        # Export to Excel
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_df.to_excel(writer, index=False, sheet_name='Combined Data')
            output.seek(0)

            st.download_button(
                label="💾 Скачать объединённый файл в Excel",
                data=output,
                file_name="combined_real_estate.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Ошибка при создании Excel файла: {str(e)}")
            
    except Exception as e:
        st.error(f"Ошибка при объединении файлов: {str(e)}")
