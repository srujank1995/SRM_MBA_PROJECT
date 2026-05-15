
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from business_conclusions import BusinessAnalyzer, AlertSeverity

# Page configuration
st.set_page_config(
    page_title="AI Demand Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .best-model {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models and encoders
@st.cache_resource
def load_all_models():
    models_dir = Path("models")
    models_dict = {}
    
    # Load individual models
    model_files = {
        'Random Forest': 'models/random_forest.pkl',
        'XGBoost': 'models/xgboost.pkl',
        'Gradient Boosting': 'models/gradient_boosting.pkl',
        'Ridge Regression': 'models/ridge_regression.pkl',
        'Linear Regression': 'models/linear_regression.pkl',
        'Neural Network': 'models/neural_network.pkl'
    }
    
    for model_name, filepath in model_files.items():
        if Path(filepath).exists():
            models_dict[model_name] = joblib.load(filepath)
    
    return models_dict

@st.cache_resource
def load_encoders():
    country_encoder = joblib.load("country_encoder.pkl")
    stock_encoder = joblib.load("stock_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return country_encoder, stock_encoder, scaler

@st.cache_data
def load_results():
    with open("model_results.json", "r") as f:
        return json.load(f)

# Load resources
models = load_all_models()
country_encoder, stock_encoder, scaler = load_encoders()
results = load_results()

# Get default model (best performing)
best_model_name = max(results, key=lambda x: results[x]['R2_Score'])

# Title
st.title("AI-BASED PRODUCT DEMAND PREDICTION FOR E-COMMERCE PLATFORMS")
st.markdown("### MBA Final Year Project - Multiple Models Framework")
st.markdown("---")

# Sidebar - Model Selection
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Select Prediction Model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
        help="Choose which trained model to use for predictions"
    )
    
    st.markdown("---")
    st.subheader("📈 Model Rankings")
    
    # Sort by R2 Score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['R2_Score'], reverse=True)
    
    for i, (model_name, metrics) in enumerate(sorted_results, 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            st.text(f"{medal} {model_name}")
        with col2:
            st.metric("R²", metrics['R2_Score'])

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Predictions", "📊 Model Comparison", "📈 Detailed Metrics", "💰 Revenue & Profit", "🎯 Business Conclusions", "ℹ️ About"])

# ==================== TAB 1: PREDICTIONS ====================
with tab1:
    st.subheader(f"Make Predictions with {selected_model}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unit_price = st.number_input(
            "Product Unit Price (£)",
            min_value=0.0,
            value=10.0,
            step=0.5,
            help="Enter the price per unit"
        )
    
    with col2:
        country = st.selectbox(
            "Select Country",
            list(country_encoder.classes_),
            help="Choose the country for the product"
        )
    
    with col3:
        stock_code = st.selectbox(
            "Select Product Stock Code",
            list(stock_encoder.classes_),
            help="Choose the product stock code"
        )
    
    # Encode inputs
    country_encoded = country_encoder.transform([country])[0]
    stock_encoded = stock_encoder.transform([stock_code])[0]
    
    # Create input dataframe
    input_data = np.array([[unit_price, country_encoded, stock_encoded]])
    
    st.markdown("---")
    
    if st.button("🔮 Predict Demand", key="predict_btn", use_container_width=True):
        # Scale data if using Neural Network
        if selected_model == 'Neural Network':
            input_data_scaled = scaler.transform(input_data)
            prediction = models[selected_model].predict(input_data_scaled)[0]
        else:
            prediction = models[selected_model].predict(input_data)[0]
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="📦 Predicted Demand",
                value=f"{round(prediction, 2)} units",
                delta=None
            )
        
        with col2:
            model_metrics = results[selected_model]
            st.metric(
                label="Model Confidence (R²)",
                value=f"{model_metrics['R2_Score']:.4f}",
                delta=f"±{model_metrics['MAE']:.2f} units (MAE)"
            )
        
        # Additional insights
        st.info(f"""
        **Prediction Details:**
        - **Model Used:** {selected_model}
        - **Unit Price:** £{unit_price}
        - **Country:** {country}
        - **Stock Code:** {stock_code}
        - **Prediction Range:** {round(prediction - model_metrics['MAE'], 2)} - {round(prediction + model_metrics['MAE'], 2)} units (±1 MAE)
        """)

# ==================== TAB 2: MODEL COMPARISON ====================
with tab2:
    st.subheader("Compare All Models Performance")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    
    # Display comparison table
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            results_df.sort_values('R2_Score', ascending=False),
            use_container_width=True,
            height=250
        )
    
    with col2:
        st.markdown("### 📊 Metrics Explained")
        st.markdown("""
        - **MAE**: Mean Absolute Error (lower is better)
        - **RMSE**: Root Mean Squared Error (lower is better)
        - **R² Score**: Coefficient of determination (higher is better)
        - **MAPE**: Mean Absolute % Error (lower is better)
        - **CV Score**: Cross-validation R² Score
        """)
    
    st.markdown("---")
    
    # R² Score Comparison Chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_r2 = go.Figure(data=[
            go.Bar(
                x=results_df.sort_values('R2_Score', ascending=True)['Model'],
                y=results_df.sort_values('R2_Score', ascending=True)['R2_Score'],
                marker_color=['#28a745' if x == max(results_df['R2_Score']) else '#007bff' 
                             for x in results_df.sort_values('R2_Score', ascending=True)['R2_Score']],
                text=results_df.sort_values('R2_Score', ascending=True)['R2_Score'],
                textposition='auto',
            )
        ])
        fig_r2.update_layout(
            title="R² Score Comparison",
            xaxis_title="Model",
            yaxis_title="R² Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        fig_mae = go.Figure(data=[
            go.Bar(
                x=results_df.sort_values('MAE', ascending=False)['Model'],
                y=results_df.sort_values('MAE', ascending=False)['MAE'],
                marker_color=['#dc3545' if x == max(results_df['MAE']) else '#ffc107' 
                             for x in results_df.sort_values('MAE', ascending=False)['MAE']],
                text=results_df.sort_values('MAE', ascending=False)['MAE'],
                textposition='auto',
            )
        ])
        fig_mae.update_layout(
            title="MAE Comparison (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="MAE",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # Error Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rmse = go.Figure(data=[
            go.Bar(
                x=results_df.sort_values('RMSE', ascending=False)['Model'],
                y=results_df.sort_values('RMSE', ascending=False)['RMSE'],
                marker_color='#6f42c1',
                text=results_df.sort_values('RMSE', ascending=False)['RMSE'],
                textposition='auto',
            )
        ])
        fig_rmse.update_layout(
            title="RMSE Comparison",
            xaxis_title="Model",
            yaxis_title="RMSE",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col2:
        fig_mape = go.Figure(data=[
            go.Bar(
                x=results_df.sort_values('MAPE', ascending=False)['Model'],
                y=results_df.sort_values('MAPE', ascending=False)['MAPE'],
                marker_color='#20c997',
                text=results_df.sort_values('MAPE', ascending=False)['MAPE'],
                textposition='auto',
            )
        ])
        fig_mape.update_layout(
            title="MAPE Comparison (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="MAPE %",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_mape, use_container_width=True)

# ==================== TAB 3: DETAILED METRICS ====================
with tab3:
    st.subheader("Detailed Model Performance Analysis")
    
    selected_metric_model = st.selectbox(
        "Select Model for Detailed View",
        list(models.keys())
    )
    
    model_info = results[selected_metric_model]
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="R² Score",
            value=f"{model_info['R2_Score']:.4f}",
            help="Coefficient of determination (1.0 is perfect)"
        )
    
    with col2:
        st.metric(
            label="MAE",
            value=f"{model_info['MAE']:.2f}",
            help="Mean Absolute Error in units"
        )
    
    with col3:
        st.metric(
            label="RMSE",
            value=f"{model_info['RMSE']:.2f}",
            help="Root Mean Squared Error"
        )
    
    with col4:
        st.metric(
            label="MAPE",
            value=f"{model_info['MAPE']:.2f}%",
            help="Mean Absolute Percentage Error"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Cross-Validation Score",
            value=f"{model_info['CV_Score']:.4f}",
            help="Average R² score from 5-fold cross-validation"
        )
    
    with col2:
        st.metric(
            label="Data Split",
            value=f"{model_info['Train_Size']} / {model_info['Test_Size']}",
            help="Training samples / Test samples"
        )
    
    st.markdown("---")
    
    # Model explanation
    explanations = {
        'Random Forest': 'Ensemble method using multiple decision trees. Good for non-linear relationships. Provides feature importance.',
        'XGBoost': 'Gradient boosting framework. Highly optimized and often best for tabular data. Fast training.',
        'Gradient Boosting': 'Sequential ensemble method. Builds trees to correct previous errors. Good generalization.',
        'Ridge Regression': 'Regularized linear model that prevents overfitting. Fast, scalable, and interpretable. Good baseline.',
        'Linear Regression': 'Baseline linear model. Simple and interpretable. Good for understanding data relationships.',
        'Neural Network': 'Deep learning approach with hidden layers. Captures complex patterns but may overfit on small datasets.'
    }
    
    st.info(f"""
    **{selected_metric_model}**
    
    {explanations.get(selected_metric_model, 'No description available')}
    """)

# ==================== TAB 4: REVENUE & PROFIT IMPACT ====================
with tab4:
    st.subheader("💰 Revenue & Profit Impact Analysis")
    st.markdown("*Make data-driven decisions based on financial impact*")
    
    # Business Parameters Section
    st.markdown("### 📋 Business Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cost_per_unit = st.number_input(
            "Product Cost per Unit (£)",
            min_value=0.0,
            value=5.0,
            step=0.5,
            help="Manufacturing or acquisition cost"
        )
    
    with col2:
        holding_cost_pct = st.slider(
            "Annual Holding Cost (%)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
            help="% of inventory value to hold per year (storage, insurance, shrinkage)"
        )
    
    with col3:
        stockout_cost_per_unit = st.number_input(
            "Stockout Cost per Unit (£)",
            min_value=0.0,
            value=10.0,
            step=0.5,
            help="Lost profit + customer dissatisfaction per unit"
        )
    
    st.markdown("---")
    
    # Revenue & Profit Analysis Section
    st.markdown("### 📊 Financial Forecast for Selected Scenario")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_unit_price = st.number_input(
            "Unit Price (£)",
            min_value=0.0,
            value=50.0,
            step=1.0,
            key="revenue_price"
        )
    
    with col2:
        forecast_country = st.selectbox(
            "Country",
            list(country_encoder.classes_),
            key="revenue_country"
        )
    
    with col3:
        forecast_stock = st.selectbox(
            "Product Stock Code",
            list(stock_encoder.classes_),
            key="revenue_stock"
        )
    
    if st.button("📈 Calculate Financial Impact", use_container_width=True):
        # Prepare input data
        country_enc = country_encoder.transform([forecast_country])[0]
        stock_enc = stock_encoder.transform([forecast_stock])[0]
        input_arr = np.array([[forecast_unit_price, country_enc, stock_enc]])
        
        # Get prediction from best model
        best_model = models[best_model_name]
        if best_model_name == 'Neural Network':
            input_scaled = scaler.transform(input_arr)
            predicted_demand = best_model.predict(input_scaled)[0]
        else:
            predicted_demand = best_model.predict(input_arr)[0]
        
        # Calculate financial metrics
        predicted_demand = max(0, predicted_demand)  # Ensure positive
        
        # Revenue calculation
        total_revenue = predicted_demand * forecast_unit_price
        
        # Cost calculations
        total_product_cost = predicted_demand * cost_per_unit
        gross_profit = total_revenue - total_product_cost
        
        # Inventory carrying cost (annual cost to hold inventory)
        avg_inventory = predicted_demand / 2  # Assuming linear consumption
        annual_holding_cost = (avg_inventory * forecast_unit_price) * (holding_cost_pct / 100)
        
        # Profit after holding costs
        net_profit_optimized = gross_profit - annual_holding_cost
        
        # Stockout scenario analysis
        stockout_cost_impact = predicted_demand * 0.1 * stockout_cost_per_unit  # 10% stockout risk
        net_profit_with_risk = net_profit_optimized - stockout_cost_impact
        
        # Profit margin
        profit_margin = (net_profit_optimized / total_revenue * 100) if total_revenue > 0 else 0
        
        # Display financial metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📦 Forecasted Demand",
                f"{round(predicted_demand, 0)} units",
                help="AI-predicted product demand"
            )
        
        with col2:
            st.metric(
                "💷 Total Revenue",
                f"£{round(total_revenue, 2)}",
                help="Revenue = Demand × Unit Price"
            )
        
        with col3:
            st.metric(
                "💚 Gross Profit",
                f"£{round(gross_profit, 2)}",
                help="Revenue - Product Costs"
            )
        
        with col4:
            st.metric(
                "📊 Profit Margin",
                f"{round(profit_margin, 2)}%",
                help="(Gross Profit / Revenue) × 100"
            )
        
        st.markdown("---")
        
        # Detailed Financial Breakdown
        st.markdown("### 💼 Detailed Financial Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create breakdown table
            breakdown_data = {
                'Metric': [
                    'Forecasted Demand',
                    'Unit Price',
                    'Product Cost/Unit',
                    '---',
                    'Total Revenue',
                    'Total Product Costs',
                    'Gross Profit',
                    'Annual Holding Cost',
                    'Net Profit (Optimized)',
                    'Stockout Risk Cost (10%)',
                    'Net Profit (With Risk)',
                ],
                'Value': [
                    f"{round(predicted_demand, 2)} units",
                    f"£{forecast_unit_price}",
                    f"£{cost_per_unit}",
                    '---',
                    f"£{round(total_revenue, 2)}",
                    f"£{round(total_product_cost, 2)}",
                    f"£{round(gross_profit, 2)}",
                    f"£{round(annual_holding_cost, 2)}",
                    f"£{round(net_profit_optimized, 2)}",
                    f"£{round(stockout_cost_impact, 2)}",
                    f"£{round(net_profit_with_risk, 2)}",
                ]
            }
            
            st.dataframe(
                pd.DataFrame(breakdown_data),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Cost breakdown pie chart
            costs_breakdown = {
                'Gross Profit': gross_profit,
                'Product Costs': total_product_cost,
                'Holding Costs': annual_holding_cost,
                'Stockout Risk': stockout_cost_impact
            }
            
            fig_costs = go.Figure(data=[go.Pie(
                labels=list(costs_breakdown.keys()),
                values=list(costs_breakdown.values()),
                hole=.3,
                marker=dict(colors=['#28a745', '#dc3545', '#ffc107', '#ff6b6b'])
            )])
            
            fig_costs.update_layout(
                title="Revenue Allocation Breakdown",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_costs, use_container_width=True)
        
        st.markdown("---")
        
        # Break-Even Analysis
        st.markdown("### 📍 Break-Even Analysis")
        
        # Initialize values for later use
        safety_margin = 0
        contribution_margin = forecast_unit_price - cost_per_unit
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate break-even point
            if contribution_margin > 0:
                # Assuming fixed costs are holding costs + operational overhead
                fixed_costs = annual_holding_cost * 1.5  # Assume 50% overhead
                breakeven_units = fixed_costs / contribution_margin
                breakeven_revenue = breakeven_units * forecast_unit_price
                
                st.metric(
                    "⏸️ Break-Even Quantity",
                    f"{round(breakeven_units, 0)} units",
                    help="Units needed to cover costs"
                )
                
                st.metric(
                    "💶 Break-Even Revenue",
                    f"£{round(breakeven_revenue, 2)}",
                    help="Revenue at break-even point"
                )
                
                # Safety margin
                safety_margin = ((predicted_demand - breakeven_units) / predicted_demand * 100) if predicted_demand > 0 else 0
                
                st.metric(
                    "🛡️ Safety Margin",
                    f"{round(max(0, safety_margin), 2)}%",
                    help="Buffer above break-even (% of forecasted demand)",
                    delta="Safe" if safety_margin > 20 else "At Risk"
                )
            else:
                st.warning("⚠️ Unit price must be higher than cost per unit for break-even analysis")
        
        with col2:
            # Break-even chart
            if contribution_margin > 0:
                breakeven_units_range = np.linspace(0, predicted_demand * 1.5, 100)
                revenue_line = breakeven_units_range * forecast_unit_price
                cost_line = (breakeven_units_range * cost_per_unit) + annual_holding_cost
                
                fig_breakeven = go.Figure()
                
                fig_breakeven.add_trace(go.Scatter(
                    x=breakeven_units_range, y=revenue_line,
                    mode='lines', name='Total Revenue',
                    line=dict(color='#28a745', width=3)
                ))
                
                fig_breakeven.add_trace(go.Scatter(
                    x=breakeven_units_range, y=cost_line,
                    mode='lines', name='Total Cost',
                    line=dict(color='#dc3545', width=3)
                ))
                
                fig_breakeven.add_vline(
                    x=predicted_demand,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"Forecasted Demand<br>{round(predicted_demand, 0)} units",
                    annotation_position="top right"
                )
                
                fig_breakeven.update_layout(
                    title="Break-Even Analysis Chart",
                    xaxis_title="Units Sold",
                    yaxis_title="Amount (£)",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_breakeven, use_container_width=True)
        
        st.markdown("---")
        
        # ROI & Efficiency Metrics
        st.markdown("### 🎯 Return on Investment (ROI) Metrics")
        
        # Calculate all metrics first (before displaying in columns)
        investment_per_unit = cost_per_unit + (forecast_unit_price * holding_cost_pct / 100)
        roi = (net_profit_optimized / investment_per_unit / predicted_demand * 100) if investment_per_unit > 0 else 0
        payback_days = (annual_holding_cost / (net_profit_optimized / 365)) if net_profit_optimized > 0 else 0
        inventory_turnover = predicted_demand / (avg_inventory if avg_inventory > 0 else 1)
        carrying_cost_ratio = (annual_holding_cost / total_revenue * 100) if total_revenue > 0 else 0
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📈 ROI",
                f"{round(max(0, roi), 2)}%",
                help="Return on investment"
            )
        
        with col2:
            st.metric(
                "📅 Payback Period",
                f"{round(max(0, payback_days), 0)} days",
                help="Days to recover inventory investment"
            )
        
        with col3:
            st.metric(
                "🔄 Inventory Turnover",
                f"{round(inventory_turnover, 2)}x",
                help="Times inventory is sold and replenished"
            )
        
        with col4:
            st.metric(
                "📦 Carrying Cost Ratio",
                f"{round(carrying_cost_ratio, 2)}%",
                help="% of revenue spent on holding inventory"
            )
        
        st.markdown("---")
        
        # Business Recommendations
        st.markdown("### 💡 Strategic Recommendations")
        
        recommendations = []
        
        if safety_margin < 10:
            recommendations.append("🔴 **High Risk**: Safety margin below 10%. Consider increasing safety stock or reducing unit price.")
        elif safety_margin < 20:
            recommendations.append("🟡 **Medium Risk**: Safety margin between 10-20%. Monitor demand closely.")
        else:
            recommendations.append("🟢 **Low Risk**: Healthy safety margin above 20%. Current forecast is reliable.")
        
        if carrying_cost_ratio > 25:
            recommendations.append("💼 **High Inventory Costs**: Consider just-in-time inventory management or bulk discounts.")
        
        if profit_margin < 10:
            recommendations.append("⚠️ **Low Profit Margin**: Consider price optimization or cost reduction strategies.")
        elif profit_margin > 30:
            recommendations.append("✅ **Healthy Profit Margin**: Strong profitability on this product.")
        
        if inventory_turnover < 4:
            recommendations.append("🐢 **Slow Turnover**: Product sells slowly. Consider promotional campaigns.")
        elif inventory_turnover > 12:
            recommendations.append("🚀 **Fast Turnover**: High-demand product. Ensure sufficient stock levels.")
        
        for rec in recommendations:
            st.info(rec)

# ==================== TAB 5: BUSINESS CONCLUSIONS ====================
with tab5:
    st.subheader("🎯 AI-Powered Business Conclusions & Recommendations")
    st.markdown("*Intelligent recommendations engine, alerts system, and business insights based on demand predictions*")
    st.markdown("---")
    
    # Input section for business analysis
    st.markdown("### 🔍 Scenario Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        biz_unit_price = st.number_input(
            "Unit Price (£)",
            min_value=0.0,
            value=25.0,
            step=0.5,
            key="biz_price"
        )
    
    with col2:
        biz_country = st.selectbox(
            "Country",
            list(country_encoder.classes_),
            key="biz_country"
        )
    
    with col3:
        biz_stock = st.selectbox(
            "Product Stock Code",
            list(stock_encoder.classes_),
            key="biz_stock"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        biz_avg_historical = st.number_input(
            "Average Historical Quantity",
            min_value=1.0,
            value=50.0,
            step=5.0,
            key="biz_historical",
            help="Average quantity sold historically for this product"
        )
    
    with col2:
        inventory_cost = st.number_input(
            "Inventory Holding Cost per Unit (£)",
            min_value=0.0,
            value=0.5,
            step=0.1,
            key="biz_inventory_cost",
            help="Cost to hold one unit in inventory per period"
        )
    
    if st.button("🚀 Generate Business Analysis", use_container_width=True, key="biz_analysis"):
        # Prepare input for prediction
        biz_country_enc = country_encoder.transform([biz_country])[0]
        biz_stock_enc = stock_encoder.transform([biz_stock])[0]
        biz_input = np.array([[biz_unit_price, biz_country_enc, biz_stock_enc]])
        
        # Get prediction from best model
        best_model = models[best_model_name]
        if best_model_name == 'Neural Network':
            biz_input_scaled = scaler.transform(biz_input)
            predicted_qty = best_model.predict(biz_input_scaled)[0]
        else:
            predicted_qty = best_model.predict(biz_input)[0]
        
        predicted_qty = max(0.1, predicted_qty)  # Ensure positive
        
        # Create analyzer
        analyzer = BusinessAnalyzer(
            unit_price=biz_unit_price,
            country=biz_country,
            stock_code=biz_stock,
            predicted_quantity=predicted_qty,
            avg_historical_quantity=biz_avg_historical,
            inventory_cost_per_unit=inventory_cost
        )
        
        # Get complete analysis
        summary = analyzer.get_summary()
        
        # ===== ALERTS SECTION =====
        st.markdown("---")
        st.markdown("### 🚨 Critical Alerts & Notifications")
        
        alerts = summary['alerts']
        if alerts:
            for alert in alerts:
                if alert.severity == AlertSeverity.CRITICAL:
                    st.error(f"{alert.severity.value}: **{alert.title}**\n\n{alert.message}\n\n**Action:** {alert.recommendation}")
                elif alert.severity == AlertSeverity.WARNING:
                    st.warning(f"{alert.severity.value}: **{alert.title}**\n\n{alert.message}\n\n**Action:** {alert.recommendation}")
                elif alert.severity == AlertSeverity.SUCCESS:
                    st.success(f"{alert.severity.value}: **{alert.title}**\n\n{alert.message}\n\n**Action:** {alert.recommendation}")
                else:
                    st.info(f"{alert.severity.value}: **{alert.title}**\n\n{alert.message}\n\n**Action:** {alert.recommendation}")
        else:
            st.info("✅ No critical alerts at this time. System operating normally.")
        
        # ===== DEMAND CLASSIFICATION & KEY METRICS =====
        st.markdown("---")
        st.markdown("### 📊 Demand Analysis Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Demand Level",
                summary['demand_level'],
                "Classification"
            )
        
        with col2:
            st.metric(
                "Prediction Confidence",
                f"{summary['confidence']*100:.0f}%",
                "Model Trust Level"
            )
        
        with col3:
            st.metric(
                "📦 Predicted Quantity",
                f"{round(summary['predicted_quantity'], 0)} units",
                f"vs {round(summary['historical_average'], 0)} historical avg"
            )
        
        with col4:
            demand_ratio = summary['predicted_quantity'] / summary['historical_average'] if summary['historical_average'] > 0 else 1
            st.metric(
                "Demand vs History",
                f"{demand_ratio:.2f}x",
                "Change Factor"
            )
        
        # ===== FINANCIAL IMPACT SECTION =====
        st.markdown("---")
        st.markdown("### 💰 Financial Impact & Metrics")
        
        financials = summary['financials']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Predicted Revenue",
                f"£{financials['predicted_revenue']:.2f}",
                f"vs £{financials['historical_revenue']:.2f} historical"
            )
        
        with col2:
            st.metric(
                "Revenue Change",
                f"£{financials['revenue_variance']:.2f}",
                f"{financials['revenue_variance_pct']:+.1f}%"
            )
        
        with col3:
            st.metric(
                "Estimated Profit",
                f"£{financials['estimated_profit']:.2f}",
                f"{financials['profit_margin_pct']:.1f}% margin"
            )
        
        with col4:
            st.metric(
                "Holding Cost",
                f"£{financials['holding_cost']:.2f}",
                f"ROI: {financials['roi']:+.0f}%"
            )
        
        # Financial breakdown visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Create financial breakdown chart
            financial_breakdown = {
                'Revenue': financials['predicted_revenue'],
                'Product Cost': financials['predicted_revenue'] * 0.35,  # Assuming 35% cost
                'Holding Cost': financials['holding_cost']
            }
            
            fig_financial = go.Figure(data=[go.Bar(
                x=list(financial_breakdown.keys()),
                y=list(financial_breakdown.values()),
                marker_color=['#28a745', '#dc3545', '#ffc107'],
                text=[f"£{v:.0f}" for v in financial_breakdown.values()],
                textposition='auto'
            )])
            
            fig_financial.update_layout(
                title="Financial Breakdown",
                yaxis_title="Amount (£)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_financial, use_container_width=True)
        
        with col2:
            # Profit margin gauge
            margin = summary['financials']['profit_margin_pct']
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=max(0, margin),
                title={'text': "Profit Margin %"},
                delta={'reference': 20},
                gauge={
                    'axis': {'range': [0, 50]},
                    'bar': {'color': '#28a745'},
                    'steps': [
                        {'range': [0, 10], 'color': '#ffcccc'},
                        {'range': [10, 20], 'color': '#ffe6cc'},
                        {'range': [20, 30], 'color': '#e6f2ff'},
                        {'range': [30, 50], 'color': '#ccffcc'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ))
            
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # ===== INVENTORY RECOMMENDATIONS =====
        st.markdown("---")
        st.markdown("### 📦 Inventory Management Recommendations")
        
        inventory = summary['inventory']
        
        inv_col1, inv_col2, inv_col3 = st.columns(3)
        
        with inv_col1:
            st.metric(
                "Safety Stock",
                f"{round(inventory['safety_stock'], 0)} units",
                "Minimum buffer"
            )
        
        with inv_col2:
            st.metric(
                "Reorder Point",
                f"{round(inventory['reorder_point'], 0)} units",
                "Trigger restocking"
            )
        
        with inv_col3:
            st.metric(
                "Max Stock Level",
                f"{round(inventory['max_stock_level'], 0)} units",
                "Optimal inventory cap"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Recommended Order Qty",
                f"{round(inventory['recommended_order_qty'], 0)} units",
                "Per order"
            )
        
        with col2:
            st.metric(
                "Stock Turnover",
                f"{round(inventory['estimated_stock_turnover'], 2)}x",
                "Times per period"
            )
        
        # ===== ACTIONABLE RECOMMENDATIONS =====
        st.markdown("---")
        st.markdown("### 🎯 Actionable Recommendations")
        
        recommendations = summary['recommendations']
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        priority_color = {
                            'HIGH': '🔴',
                            'MEDIUM': '🟡',
                            'LOW': '🟢'
                        }.get(rec.priority, '⚪')
                        st.markdown(f"**{priority_color} {rec.priority}**")
                    
                    with col2:
                        st.markdown(f"**{rec.action}**")
                        st.markdown(f"Impact: {rec.impact}")
                        st.markdown(f"Details: {rec.details}")
                    
                    st.divider()
        else:
            st.info("No specific recommendations at this time.")
        
        # ===== SUMMARY TABLE =====
        st.markdown("---")
        st.markdown("### 📋 Complete Analysis Summary")
        
        summary_table = {
            'Metric': [
                'Product Code',
                'Country',
                'Unit Price',
                'Predicted Quantity',
                'Historical Average',
                'Demand Level',
                'Confidence Score',
                'Predicted Revenue',
                'Estimated Profit',
                'Profit Margin',
                'ROI',
                'Safety Stock',
                'Reorder Point',
                'Recommended Order Qty'
            ],
            'Value': [
                summary['product_code'],
                summary['country'],
                f"£{summary['unit_price']:.2f}",
                f"{round(summary['predicted_quantity'], 2)} units",
                f"{round(summary['historical_average'], 2)} units",
                summary['demand_level'],
                f"{summary['confidence']*100:.0f}%",
                f"£{financials['predicted_revenue']:.2f}",
                f"£{financials['estimated_profit']:.2f}",
                f"{financials['profit_margin_pct']:.2f}%",
                f"{financials['roi']:.2f}%",
                f"{round(inventory['safety_stock'], 2)} units",
                f"{round(inventory['reorder_point'], 2)} units",
                f"{round(inventory['recommended_order_qty'], 2)} units"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(summary_table),
            use_container_width=True,
            hide_index=True
        )

# ==================== TAB 6: ABOUT ====================
with tab6:
    st.subheader("📚 Project Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Project Overview
        **AI-Based Product Demand Prediction for E-Commerce Platforms**
        
        This project demonstrates advanced machine learning techniques for predicting product demand in e-commerce businesses.
        
        ### Key Features
        - ✅ Multiple Model Comparison
        - ✅ Cross-Validation Analysis
        - ✅ Comprehensive Metrics Evaluation
        - ✅ Interactive Predictions
        - ✅ Model Performance Dashboard
        """)
    
    with col2:
        st.markdown("""
        ### Technologies Used
        - **Python** - Programming language
        - **Scikit-learn** - ML algorithms
        - **XGBoost** - Advanced gradient boosting
        - **Streamlit** - Interactive dashboard
        - **Plotly** - Interactive visualizations
        - **Pandas & NumPy** - Data manipulation
        
        ### Models Included
        1. Random Forest
        2. XGBoost
        3. Gradient Boosting
        4. Ridge Regression
        5. Linear Regression
        6. Neural Network (MLPRegressor)
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 How to Use This Application
    
    1. **Make Predictions Tab**: Select input values (price, country, stock code) and get demand predictions using your chosen model
    
    2. **Model Comparison Tab**: View side-by-side comparison of all models with different evaluation metrics
    
    3. **Detailed Metrics Tab**: Explore detailed performance metrics for each model
    
    4. **Revenue & Profit Tab**: Analyze financial impact including revenue forecasts, profit margins, break-even analysis, and ROI metrics
    
    ### 🎯 Model Selection Guide
    - **Best Overall**: Choose the model ranked #1 (highest R² score)
    - **Fastest Inference**: Linear Regression or Random Forest
    - **Most Accurate**: Usually XGBoost or Gradient Boosting
    - **Best for Production**: XGBoost (balance of speed and accuracy)
    """)
    
    st.markdown("---")
    st.markdown("**MBA Final Year Project** | Data Science & AI Stream | 2024-2026")

