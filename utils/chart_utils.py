import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

def create_dynamic_chart(df, chart_config):
    # Create charts based on users' config
    chart_type = chart_config['type']
    
    try:
        if chart_type == 'Bar Chart':
            if chart_config.get('y_axis') and chart_config.get('x_axis'):
                if chart_config.get('aggregation') == 'count':
                    data = df.groupby(chart_config['x_axis']).size().reset_index(name='count')
                    fig = px.bar(data, x=chart_config['x_axis'], y='count', 
                               title=chart_config.get('title', 'Bar Chart'),
                               color='count' if chart_config.get('color_by_value') else None)
                else:
                    agg_func = chart_config.get('aggregation', 'sum')
                    data = df.groupby(chart_config['x_axis'])[chart_config['y_axis']].agg(agg_func).reset_index()
                    fig = px.bar(data, x=chart_config['x_axis'], y=chart_config['y_axis'],
                               title=chart_config.get('title', 'Bar Chart'))
        
        elif chart_type == 'Line Chart':
            if chart_config.get('x_axis') and chart_config.get('y_axis'):
                if pd.api.types.is_datetime64_any_dtype(df[chart_config['x_axis']]):
                    freq = chart_config.get('time_freq', 'D')
                    data = df.groupby(pd.Grouper(key=chart_config['x_axis'], freq=freq))[chart_config['y_axis']].sum().reset_index()
                else:
                    data = df.groupby(chart_config['x_axis'])[chart_config['y_axis']].sum().reset_index()
                fig = px.line(data, x=chart_config['x_axis'], y=chart_config['y_axis'],
                            title=chart_config.get('title', 'Line Chart'), markers=True)
        
        elif chart_type == 'Scatter Plot':
            if chart_config.get('x_axis') and chart_config.get('y_axis'):
                color_col = chart_config.get('color_column') if chart_config.get('color_column') != 'None' else None
                fig = px.scatter(df.sample(min(10000, len(df))), 
                               x=chart_config['x_axis'], y=chart_config['y_axis'],
                               color=color_col, size=None,
                               title=chart_config.get('title', 'Scatter Plot'),
                               opacity=0.7)
        
        elif chart_type == 'Pie Chart':
            if chart_config.get('category_column'):
                if chart_config.get('value_column') and chart_config['value_column'] != 'count':
                    data = df.groupby(chart_config['category_column'])[chart_config['value_column']].sum().reset_index()
                    fig = px.pie(data, names=chart_config['category_column'], values=chart_config['value_column'],
                               title=chart_config.get('title', 'Pie Chart'))
                else:
                    data = df[chart_config['category_column']].value_counts().reset_index()
                    fig = px.pie(data, names='index', values=chart_config['category_column'],
                               title=chart_config.get('title', 'Pie Chart'))
        
        elif chart_type == 'Box Plot':
            if chart_config.get('y_axis'):
                x_col = chart_config.get('x_axis') if chart_config.get('x_axis') != 'None' else None
                fig = px.box(df, x=x_col, y=chart_config['y_axis'],
                           title=chart_config.get('title', 'Box Plot'))
        
        elif chart_type == 'Histogram':
            if chart_config.get('column'):
                bins = chart_config.get('bins', 30)
                fig = px.histogram(df, x=chart_config['column'], nbins=bins,
                                 title=chart_config.get('title', 'Histogram'))
        
        elif chart_type == 'Heatmap':
            if chart_config.get('x_axis') and chart_config.get('y_axis') and chart_config.get('value_column'):
                data = df.groupby([chart_config['x_axis'], chart_config['y_axis']])[chart_config['value_column']].sum().unstack(fill_value=0)
                fig = px.imshow(data, title=chart_config.get('title', 'Heatmap'))
        
        else:
            return None
            
        # Apply common styling
        fig.update_layout(
            height=500,
            showlegend=True,
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def chart_builder_ui(df, chart_id):
    """UI for building a custom chart; returns (config, submitted)"""
    with st.form(f"chart_form_{chart_id}", clear_on_submit=False):
        st.subheader(f"📊 Chart Builder #{chart_id + 1}")

        # Column info
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        all_cols = df.columns.tolist()

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            chart_type = st.selectbox(
                "Chart Type", 
                ['Bar Chart', 'Line Chart', 'Scatter Plot', 'Pie Chart', 'Box Plot', 
                 'Histogram', 'Heatmap'],
                key=f"chart_type_{chart_id}"
            )
        with col2:
            title = st.text_input("Chart Title", value=f"Custom {chart_type}", key=f"title_{chart_id}")
        with col3:
            submitted = st.form_submit_button(f"Generate Chart #{chart_id + 1}", use_container_width=True)

        # Build config based on chart type
        config = {'type': chart_type, 'title': title, 'plot_bgcolor': 'rgba(0, 0, 0, 0)'}

        if chart_type in ['Bar Chart', 'Line Chart']:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                config['x_axis'] = st.selectbox("X-Axis", all_cols, key=f"x_{chart_id}")
            with col_b:
                config['y_axis'] = st.selectbox("Y-Axis", numeric_cols, key=f"y_{chart_id}")
            with col_c:
                config['aggregation'] = st.selectbox("Aggregation", ['sum', 'mean', 'count', 'max', 'min'], key=f"agg_{chart_id}")
            if chart_type == 'Line Chart' and config['x_axis'] in datetime_cols:
                config['time_freq'] = st.selectbox("Time Frequency", ['D', 'W', 'M', 'Q', 'Y'], key=f"freq_{chart_id}")

        elif chart_type == 'Scatter Plot':
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                config['x_axis'] = st.selectbox("X-Axis", numeric_cols, key=f"scatter_x_{chart_id}")
            with col_b:
                config['y_axis'] = st.selectbox("Y-Axis", numeric_cols, key=f"scatter_y_{chart_id}")
            with col_c:
                config['color_column'] = st.selectbox("Color By", ['None'] + categorical_cols, key=f"color_{chart_id}")
            # with col_d:
            #     config['size_column'] = st.selectbox("Size By", ['None'] + numeric_cols, key=f"size_{chart_id}")

        elif chart_type == 'Pie Chart':
            col_a, col_b = st.columns(2)
            with col_a:
                config['category_column'] = st.selectbox("Category", categorical_cols, key=f"pie_cat_{chart_id}")
            with col_b:
                config['value_column'] = st.selectbox("Value", numeric_cols, key=f"pie_val_{chart_id}")

        elif chart_type == 'Box Plot':
            col_a, col_b = st.columns(2)
            with col_a:
                config['y_axis'] = st.selectbox("Y-Axis (Numeric)", numeric_cols, key=f"box_y_{chart_id}")
            with col_b:
                config['x_axis'] = st.selectbox("Group By", ['None'] + categorical_cols, key=f"box_x_{chart_id}")

        elif chart_type == 'Histogram':
            col_a, col_b = st.columns(2)
            with col_a:
                config['column'] = st.selectbox("Column", numeric_cols, key=f"hist_col_{chart_id}")
            with col_b:
                config['bins'] = st.slider("Number of Bins", 10, 100, 30, key=f"bins_{chart_id}")

        elif chart_type == 'Heatmap':
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                config['x_axis'] = st.selectbox("X-Axis", categorical_cols, key=f"heat_x_{chart_id}")
            with col_b:
                config['y_axis'] = st.selectbox("Y-Axis", categorical_cols, key=f"heat_y_{chart_id}")
            with col_c:
                config['value_column'] = st.selectbox("Values", numeric_cols, key=f"heat_val_{chart_id}")

        return config, submitted
