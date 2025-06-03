import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Шаблон стандартных колонок
STANDARD_COLUMNS = {
    "Комплекс": ["Комплекс", "Project", "Name", "SALES"],
    "Район": ["Район", "District", "Area", "Location"],
    "Площадь": ["Площадь", "Площадь (м²)", "Area (sqm)", "Area", "SQM"],
    "Спален": ["Спален", "Bedrooms", "Beds", "Bed"],
    "Этажей": ["Этажей", "Floors", "Floor"],
    "Цена": ["Price, ฿", "Цена", "Price (THB)", "Price", "THB"]
}

def detect_header_row(df, max_rows_to_check=5):
    """Определяет наиболее вероятную строку заголовков"""
    preview_rows = []
    best_header_score = 0
    suggested_header_row = 0

    for i in range(min(max_rows_to_check, len(df))):
        row = df.iloc[i]
        non_empty = row.count()
        unnamed_count = sum(1 for col in row.index if 'Unnamed' in str(col))
        
        # Calculate header likelihood score
        score = 0
        row_values = [str(val).strip() for val in row if pd.notna(val)]
        
        # Characteristics of a header row:
        # 1. High number of non-empty cells
        score += non_empty * 2
        
        # 2. Low number of numeric values (headers are usually text)
        numeric_count = sum(1 for val in row_values if str(val).replace('.', '').isdigit())
        score -= numeric_count * 3
        
        # 3. No very long text (headers are usually short)
        long_text_count = sum(1 for val in row_values if len(str(val)) > 50)
        score -= long_text_count * 2
        
        # 4. Common header keywords
        header_keywords = ['project', 'type', 'size', 'area', 'price', 'location', 'floor', 'room', 'bed', 'bath', 'contact']
        keyword_matches = sum(1 for val in row_values if any(keyword.lower() in str(val).lower() for keyword in header_keywords))
        score += keyword_matches * 3
        
        # 5. Penalize rows with URLs or contact information
        url_count = sum(1 for val in row_values if 'http' in str(val).lower() or 'www' in str(val).lower())
        phone_count = sum(1 for val in row_values if any(char.isdigit() for char in str(val)) and '+' in str(val))
        score -= (url_count + phone_count) * 5
        
        # Store row information
        row_info = {
            'row_num': i,
            'content': row.tolist(),
            'non_empty': non_empty,
            'unnamed_count': unnamed_count,
            'score': score,
            'header_likelihood': 'Низкая'
        }
        
        # Update best score
        if score > best_header_score:
            best_header_score = score
            suggested_header_row = i
            
        preview_rows.append(row_info)
    
    # Update likelihood labels
    for row in preview_rows:
        if row['row_num'] == suggested_header_row:
            row['header_likelihood'] = 'Высокая'
        elif row['score'] > best_header_score * 0.7:
            row['header_likelihood'] = 'Средняя'
    
    return preview_rows, suggested_header_row

def analyze_prices(df):
    """Анализ цен по районам"""
    if 'Цена' not in df.columns or 'Район' not in df.columns or 'Площадь' not in df.columns:
        return None, None, None

    # Очистка и преобразование данных
    df = df.copy()
    df['Цена'] = pd.to_numeric(df['Цена'].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
    df['Площадь'] = pd.to_numeric(df['Площадь'].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
    
    # Расчет цены за квадратный метр
    df['Цена_за_м2'] = df['Цена'] / df['Площадь']
    
    # Расчет средней цены по районам
    avg_price_by_district = df.groupby('Район')['Цена_за_м2'].mean()
    
    # Поиск объектов с ценой ниже 80% от средней по району
    df['Средняя_цена_района'] = df['Район'].map(avg_price_by_district)
    df['Процент_от_средней'] = (df['Цена_за_м2'] / df['Средняя_цена_района']) * 100
    good_deals = df[df['Процент_от_средней'] < 80].copy()
    
    # Сортировка по выгодности (процент от средней цены)
    good_deals = good_deals.sort_values('Процент_от_средней')
    
    return df, good_deals, avg_price_by_district

# Настройка страницы
st.set_page_config(
    page_title="Импорт и анализ недвижимости",
    page_icon="🏠",
    layout="wide"
)

# Заголовок с логотипом
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://kvaris.com/wp-content/uploads/2.svg", width=100)
with col2:
    st.title("Импорт и анализ данных недвижимости")

# Загрузка файлов
uploaded_files = st.file_uploader(
    "Загрузите файлы Excel или CSV с данными о недвижимости",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    all_data = []
    
    for file in uploaded_files:
        st.subheader(f"📄 Обработка файла: {file.name}")
        
        try:
            # Чтение файла без указания заголовков
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, header=None)
            else:
                excel_file = pd.ExcelFile(file)
                sheets = excel_file.sheet_names
                
                if len(sheets) > 1:
                    selected_sheet = st.selectbox(
                        "Выберите лист Excel:",
                        sheets,
                        key=f"sheet_select_{file.name}"
                    )
                    df = pd.read_excel(file, sheet_name=selected_sheet, header=None)
                else:
                    df = pd.read_excel(file, header=None)

            # Анализ первых строк файла
            preview_rows, suggested_header = detect_header_row(df)
            
            # Показываем превью данных
            st.write("👀 Превью первых строк файла:")
            for row in preview_rows:
                st.write(
                    f"Строка {row['row_num'] + 1}: {row['content']}\n"
                    f"Непустых ячеек: {row['non_empty']}, "
                    f"Вероятность заголовка: {row['header_likelihood']}"
                )

            # Выбор строки заголовков с предварительно выбранным значением
            header_row = st.number_input(
                "Выберите номер строки с заголовками:",
                min_value=1,
                max_value=len(df),
                value=suggested_header + 1,
                key=f"header_select_{file.name}"
            ) - 1

            # Применяем выбранную строку как заголовок
            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:].copy()
            
            # Очистка данных
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed:', na=False)]
            
            # Показать найденные колонки
            valid_columns = [str(col) for col in df.columns if not pd.isna(col) and str(col).strip() != ""]
            st.write("🔎 Найденные колонки:", valid_columns)
            
            # Сопоставление колонок
            column_map = {}
            for std_col, possible_names in STANDARD_COLUMNS.items():
                # Автоматический поиск подходящей колонки
                found_col = None
                for name in possible_names:
                    matching_cols = [col for col in valid_columns if name.lower() in str(col).lower()]
                    if matching_cols:
                        found_col = matching_cols[0]
                        break
                
                # Выбор колонки пользователем
                column_map[std_col] = st.selectbox(
                    f"Выберите колонку для '{std_col}'",
                    options=[""] + valid_columns,
                    index=valid_columns.index(found_col) + 1 if found_col else 0,
                    key=f"{file.name}_{std_col}"
                )
            
            # Создание нового DataFrame с выбранными колонками
            mapped_df = pd.DataFrame()
            for std_col, selected_col in column_map.items():
                if selected_col:
                    mapped_df[std_col] = df[selected_col]
            
            if not mapped_df.empty:
                all_data.append(mapped_df)
                st.success(f"✅ Файл {file.name} обработан успешно")
            else:
                st.warning("⚠️ Не выбрано ни одной колонки для импорта")
            
        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла {file.name}: {str(e)}")
            continue
    
    if all_data:
        # Объединение всех данных
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Анализ данных
        df_analyzed, good_deals, avg_prices = analyze_prices(final_df)
        
        if df_analyzed is not None:
            # Вкладки для разных видов анализа
            tab1, tab2, tab3 = st.tabs(["📊 Все данные", "💰 Выгодные предложения", "📈 Анализ цен"])
            
            with tab1:
                st.subheader("Все объекты")
                st.dataframe(final_df)
            
            with tab2:
                st.subheader("🎯 Выгодные предложения (ниже 80% средней цены района)")
                if not good_deals.empty:
                    # Форматирование для отображения
                    display_deals = good_deals.copy()
                    display_deals['Цена_за_м2'] = display_deals['Цена_за_м2'].round(0).astype(int)
                    display_deals['Процент_от_средней'] = display_deals['Процент_от_средней'].round(1)
                    display_deals['Экономия'] = (100 - display_deals['Процент_от_средней']).round(1)
                    
                    # Отображение результатов
                    for _, row in display_deals.iterrows():
                        with st.container():
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"""
                                    **{row['Комплекс']}** в районе {row['Район']}
                                    - 💰 Цена: ฿{int(row['Цена']):,}
                                    - 📏 Площадь: {int(row['Площадь'])} м²
                                    - 🏷️ Цена за м²: ฿{int(row['Цена_за_м2']):,}
                                    """)
                            with col2:
                                st.markdown(f"""
                                    **Выгода: {row['Экономия']}%**
                                    - 🎯 {row['Процент_от_средней']}% от средней
                                    - 🛏️ Спален: {row['Спален']}
                                    - 🏗️ Этажей: {row['Этажей']}
                                    """)
                            st.divider()
                else:
                    st.info("Не найдено предложений ниже 80% от средней цены района")
            
            with tab3:
                st.subheader("Анализ цен по районам")
                
                # График средних цен по районам
                fig, ax = plt.subplots(figsize=(10, 6))
                avg_prices.sort_values().plot(kind='barh', ax=ax)
                plt.title("Средняя цена за м² по районам")
                plt.xlabel("Цена за м² (฿)")
                st.pyplot(fig)
                
                # Статистика по районам
                st.subheader("Статистика по районам")
                stats_df = df_analyzed.groupby('Район').agg({
                    'Цена_за_м2': ['mean', 'min', 'max', 'count']
                }).round(0)
                stats_df.columns = ['Средняя цена/м²', 'Мин цена/м²', 'Макс цена/м²', 'Количество объектов']
                st.dataframe(stats_df)
        
        # Подготовка файла для скачивания
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_df.to_excel(writer, sheet_name='Все объекты', index=False)
            if df_analyzed is not None and not good_deals.empty:
                good_deals.to_excel(writer, sheet_name='Выгодные предложения', index=False)
        output.seek(0)
        
        # Кнопка скачивания
        st.download_button(
            label="💾 Скачать данные (включая анализ)",
            data=output,
            file_name="real_estate_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ) 