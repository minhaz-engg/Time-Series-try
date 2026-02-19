import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from prophet import Prophet 
import warnings
import logging

# ==========================================
# 0. High-Performance Configuration
# ==========================================
st.set_page_config(
    page_title="Supply Chain AI Cortex | Enterprise Edition",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)
warnings.filterwarnings('ignore')

# Suppress Prophet's verbose logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# Custom CSS for "Dark Mode" Professional Look
st.markdown("""
<style>
    .stMetric {
        background-color: #0E1117;
        border: 1px solid #30333F;
        padding: 15px;
        border-radius: 5px;
    }
    .stProgress .st-bo {
        background-color: #00AA00;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Data Pipeline
# ==========================================
@st.cache_data
def load_lean_data(filepath='m5_lean.csv'):
    """Loads and caches the optimized dataset."""
    try:
        df = pd.read_csv(filepath)
        
        # --- FIX: ROBUST DATE RECONSTRUCTION ---
        if 'd_num' not in df.columns:
            df['d_num'] = df['d'].apply(lambda x: int(x.split('_')[1]))

        if 'date' not in df.columns:
            start_date = pd.Timestamp("2011-01-29")
            df['date'] = start_date + pd.to_timedelta(df['d_num'] - 1, unit='D')
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        return df
    except FileNotFoundError:
        return None

def create_features(df):
    """
    Rigorously regenerates temporal features for Tree Models (LGBM/XGB/CatBoost).
    """
    df = df.copy()
    
    # 1. Autoregressive Lags
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['lag_14'] = df['sales'].shift(14)
    df['lag_28'] = df['sales'].shift(28)
    
    # 2. Rolling Statistical Windows
    df['rolling_mean_7'] = df['sales'].shift(1).rolling(7).mean()
    df['rolling_std_7'] = df['sales'].shift(1).rolling(7).std()
    df['rolling_mean_28'] = df['sales'].shift(1).rolling(28).mean()
    
    # 3. Price Elasticity Signals
    df['price_change'] = df['sell_price'].pct_change()
    df['price_volatility'] = df['sell_price'].rolling(7).std()
    
    return df

def prepare_training_data(df, item_id, store_id):
    """Filters and encodes data for a specific time-series."""
    data = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
    
    # Robust Label Encoding
    le = LabelEncoder()
    data['event_name_1'] = data['event_name_1'].astype(str).fillna('None')
    data['event_name_1'] = le.fit_transform(data['event_name_1'])
    # Convert to numeric for models
    data['event_name_1'] = data['event_name_1'].astype(int) 
    
    return data

# ==========================================
# 2. Advanced Model Architecture
# ==========================================
def train_lightgbm(X_train, y_train, X_val, y_val):
    params = {
        'objective': 'tweedie', 
        'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'learning_rate': 0.03, 
        'num_leaves': 64,       
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'lambda_l1': 0.1,       
        'lambda_l2': 0.1,       
        'n_jobs': -1,
        'verbosity': -1
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    model = lgb.train(
        params, train_set, num_boost_round=1500,
        valid_sets=[train_set, val_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    # Ensure y is float and clean
    y_train = y_train.astype(float)
    y_val = y_val.astype(float)

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    
    params = {
        'objective': 'reg:tweedie',
        'tweedie_variance_power': 1.1,
        'eval_metric': 'rmse',
        'eta': 0.03,
        'max_depth': 8,        
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'alpha': 0.1,          
        'lambda': 0.1,         
        'nthread': -1,
        'verbosity': 0
    }
    
    model = xgb.train(
        params, dtrain, num_boost_round=1500,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    return model

def train_catboost(X_train, y_train, X_val, y_val, features):
    # Identify categorical features dynamically
    cat_cols = ['event_name_1', 'snap_CA', 'snap_TX', 'snap_WI']
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Tweedie:variance_power=1.1',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=100,
        verbose=0,
        allow_writing_files=False
    )
    
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model

def train_prophet_model(df_train):
    """Trains Prophet. Requires columns: date -> ds, sales -> y, regressors."""
    prophet_df = df_train[['date', 'sales', 'sell_price']].copy()
    prophet_df.columns = ['ds', 'y', 'sell_price']
    
    model = Prophet(
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative',
        interval_width=0.95
    )
    model.add_regressor('sell_price')
    
    model.fit(prophet_df)
    return model

# ==========================================
# 3. Simulation Engines (Recursive & Batch)
# ==========================================
def recursive_predict(model, df, start_day, model_type):
    """
    Recursive Engine for Tree Models (LGBM, XGB, CatBoost).
    """
    df = create_features(df)
    
    exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                    'sales', 'd', 'date', 'wm_yr_wk', 'd_num']
    
    features = [c for c in df.columns if c not in exclude_cols]
    future_horizon = 28
    
    progress_text = "Running Recursive Stochastic Simulation..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, day in enumerate(range(start_day, start_day + future_horizon)):
        # 1. Update features based on previous day's predictions
        df = create_features(df)
        
        mask = df['d_num'] == day
        X_test = df[mask][features]
        
        if X_test.empty: break
            
        # 2. Inference
        if model_type == 'LightGBM':
            pred = model.predict(X_test)[0]
        elif model_type == 'XGBoost':
            dtest = xgb.DMatrix(X_test, enable_categorical=True)
            pred = model.predict(dtest)[0]
        elif model_type == 'CatBoost':
            pred = model.predict(X_test)[0]
        
        # 3. Update Sales for recursion
        pred = max(0, pred) 
        df.loc[mask, 'sales'] = pred
        
        my_bar.progress((i + 1) / future_horizon, text=f"Simulating Day {day}...")
        
    my_bar.empty()
    return df, features

def batch_predict_prophet(model, grid, start_day):
    """Batch Engine for Prophet."""
    future_mask = grid['d_num'] >= start_day
    future_df = grid[future_mask][['date', 'sell_price']].copy()
    future_df.columns = ['ds', 'sell_price']
    
    forecast = model.predict(future_df)
    grid.loc[future_mask, 'sales'] = forecast['yhat'].values
    
    return grid, ['sell_price', 'trend', 'weekly', 'yearly']

def calculate_inventory_plan(df, start_day, rmse, lead_time, service_level_z, initial_stock):
    """Advanced Inventory Logic"""
    plan = df[df['d_num'] >= start_day].copy()
    
    plan['safety_stock'] = service_level_z * rmse * np.sqrt(lead_time)
    
    inventory_levels = []
    order_quantities = []
    stock = initial_stock
    
    for _, row in plan.iterrows():
        # Morning Check
        reorder_point = row['safety_stock'] + (row['sales'] * lead_time)
        
        order_qty = 0
        if stock < reorder_point:
            # Order to cover Lead Time + 7 Days Demand + Safety Stock
            target_level = reorder_point + (row['sales'] * 7)
            order_qty = target_level - stock
            stock += order_qty 
            
        # Evening Demand
        stock -= row['sales']
        
        inventory_levels.append(max(0, stock))
        order_quantities.append(max(0, order_qty))
        
    plan['projected_inventory'] = inventory_levels
    plan['recommended_order'] = order_quantities
    
    return plan

# ==========================================
# 4. Interactive Visualization (Plotly)
# ==========================================
def plot_interactive_lifecycle(history, forecast, item, rmse):
    history_view = history[history['d_num'] > (1913 - 120)]
    
    fig = go.Figure()
    
    x_hist = history_view['date'] if 'date' in history_view.columns else history_view['d_num']
    x_fore = forecast['date'] if 'date' in forecast.columns else forecast['d_num']
    
    # Historical
    fig.add_trace(go.Scatter(
        x=x_hist, y=history_view['sales'],
        mode='lines', name='Historical Sales',
        line=dict(color='gray', width=1.5), opacity=0.6
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=x_fore, y=forecast['sales'],
        mode='lines+markers', name='AI Forecast',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=5)
    ))
    
    # Confidence Interval
    upper_bound = forecast['sales'] + (1.96 * rmse)
    lower_bound = forecast['sales'] - (1.96 * rmse)
    lower_bound = lower_bound.apply(lambda x: max(0, x))
    
    fig.add_trace(go.Scatter(
        x=pd.concat([x_fore, x_fore[::-1]]),
        y=pd.concat([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 204, 150, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title=f"<b>Deep Learning Trajectory: {item}</b>",
        xaxis_title="Date",
        yaxis_title="Sales Volume",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_interactive_action_plan(plan, item):
    """
    Advanced Inventory Action Plan Plot
    - Highlights Safety Stock Threshold (Red Zone)
    - Explicitly marks Reorder Events (Green Bars)
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    x_axis = plan['date'] if 'date' in plan.columns else plan['d_num']
    
    # 1. Safety Stock Zone
    fig.add_trace(
        go.Scatter(
            x=x_axis, 
            y=plan['safety_stock'], 
            name="Safety Stock (Risk Threshold)",
            line=dict(color='rgba(255, 99, 71, 0.5)', width=1, dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(255, 99, 71, 0.1)', 
            hoverinfo='skip'
        ),
        secondary_y=False
    )

    # 2. Predicted Demand
    fig.add_trace(
        go.Scatter(
            x=x_axis, 
            y=plan['sales'], 
            name="Predicted Daily Demand",
            line=dict(color='#A0A0A0', width=1.5, dash='dot'), 
            opacity=0.6
        ),
        secondary_y=False
    )
    
    # 3. Projected Inventory Level
    fig.add_trace(
        go.Scatter(
            x=x_axis, 
            y=plan['projected_inventory'], 
            name="Projected Stock Level",
            line=dict(color='#1F77B4', width=3),
            fill='tonexty', 
            fillcolor='rgba(31, 119, 180, 0.05)'
        ),
        secondary_y=False
    )
    
    # 4. Reorder Events
    orders = plan[plan['recommended_order'] > 0]
    if not orders.empty:
        x_orders = orders['date'] if 'date' in orders.columns else orders['d_num']
        
        fig.add_trace(
            go.Bar(
                x=x_orders, 
                y=orders['recommended_order'], 
                name="Replenishment Order (Qty)",
                marker_color='#2CA02C',
                opacity=0.9,
                text=orders['recommended_order'].astype(int), 
                textposition='auto'
            ),
            secondary_y=True
        )

    fig.update_layout(
        title=dict(text=f"<b>🛡️ Inventory Strategy & Replenishment Plan: {item}</b>", font=dict(size=20)),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.5)"),
        height=550 
    )
    
    fig.update_yaxes(title_text="<b>Stock Level (Units)</b>", secondary_y=False, showgrid=True, gridcolor='#F0F0F0')
    max_order = plan['recommended_order'].max() if not plan['recommended_order'].empty else 10
    fig.update_yaxes(title_text="<b>Order Quantity</b>", secondary_y=True, showgrid=False, range=[0, max_order * 1.5])
    
    return fig

def plot_feature_importance(model, features, model_type):
    if model_type == 'Prophet':
        return None
        
    if model_type == 'LightGBM':
        importance = model.feature_importance(importance_type='gain')
    elif model_type == 'XGBoost':
        importance_map = model.get_score(importance_type='gain')
        importance = [importance_map.get(f, 0) for f in features]
    elif model_type == 'CatBoost':
        importance = model.get_feature_importance()
        
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(15)
    
    fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                 title='<b>Model Explainability: Top Drivers</b>',
                 color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(template="plotly_white")
    return fig

# ==========================================
# 5. Main Application Logic
# ==========================================
def main():
    st.title("🧠 Supply Chain AI Cortex")
    st.markdown("### Next-Gen Stochastic Forecasting & Replenishment System")
    
    data = load_lean_data()
    if data is None:
        st.error("⚠️ System Offline: `m5_lean.csv` not detected. Please execute the data engineering pipeline.")
        return

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Target Selection")
        selected_store = st.selectbox("Store Location", data['store_id'].unique())
        selected_item = st.selectbox("Product SKU", data['item_id'].unique())
        
        st.subheader("Model Hyperparameters")
        # Added CatBoost to options
        selected_model = st.radio("AI Architecture", ["LightGBM", "XGBoost", "CatBoost", "Prophet"], index=0)
        
        st.markdown("---")
        st.subheader("Inventory Simulation Parameters")
        lead_time = st.slider("Supplier Lead Time (Days)", 1, 14, 3)
        service_level = st.slider("Target Service Level (%)", 80, 99, 95)
        current_stock = st.number_input("Current Stock On-Hand", min_value=0, value=50)
        
        z_score_map = {99: 2.33, 95: 1.64, 90: 1.28, 85: 1.04, 80: 0.84}
        z_score = z_score_map.get(service_level, 1.64)

    # --- Main Block ---
    if st.button("🚀 Run AI Analysis", type="primary"):
        
        with st.status("Initializing AI Cortex...", expanded=True) as status:
            st.write("🔧 Preparing Data Grid...")
            grid = prepare_training_data(data, selected_item, selected_store)
            
            SPLIT_DAY = 1913
            
            # --- BRANCHING LOGIC FOR MODELS ---
            if selected_model == "Prophet":
                st.write(f"🧠 Training {selected_model} (Bayesian Time Series)...")
                
                train_mask = grid['d_num'] <= SPLIT_DAY
                train_data = grid[train_mask]
                
                model = train_prophet_model(train_data)
                
                val_mask = (grid['d_num'] > SPLIT_DAY - 28) & (grid['d_num'] <= SPLIT_DAY)
                val_data = grid[val_mask][['date', 'sell_price']].copy()
                val_data.columns = ['ds', 'sell_price']
                val_preds = model.predict(val_data)['yhat'].values
                y_val = grid[val_mask]['sales'].values
                
                rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                
                st.write("🔮 Generating Batch Forecast (28 Days Horizon)...")
                forecast_grid, final_features = batch_predict_prophet(model, grid.copy(), 1914)

            else:
                # Tree Models (LightGBM / XGBoost / CatBoost)
                st.write("🔧 Engineering temporal features (Lags, Rolling Windows)...")
                
                grid_feat = create_features(grid)
                grid_feat = grid_feat.dropna(subset=['lag_28', 'rolling_mean_28'])
                
                # Fatal Error Fix: XGBoost/CatBoost demand clean targets
                grid_feat = grid_feat.dropna(subset=['sales'])

                exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                                'sales', 'd', 'date', 'wm_yr_wk', 'd_num']
                features = [c for c in grid_feat.columns if c not in exclude_cols]
                
                train_mask = grid_feat['d_num'] <= SPLIT_DAY - 28
                val_mask = (grid_feat['d_num'] > SPLIT_DAY - 28) & (grid_feat['d_num'] <= SPLIT_DAY)
                
                X_train = grid_feat[train_mask][features]
                y_train = grid_feat[train_mask]['sales']
                X_val = grid_feat[val_mask][features]
                y_val = grid_feat[val_mask]['sales']
                
                st.write(f"🧠 Training {selected_model} (Tweedie Loss Optimization)...")
                if selected_model == "LightGBM":
                    model = train_lightgbm(X_train, y_train, X_val, y_val)
                    val_preds = model.predict(X_val)
                elif selected_model == "XGBoost":
                    model = train_xgboost(X_train, y_train, X_val, y_val)
                    dval = xgb.DMatrix(X_val, enable_categorical=True)
                    val_preds = model.predict(dval)
                elif selected_model == "CatBoost":
                    model = train_catboost(X_train, y_train, X_val, y_val, features)
                    val_preds = model.predict(X_val)
                
                rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                
                st.write("🔮 Generating Recursive Forecast (28 Days Horizon)...")
                forecast_grid, final_features = recursive_predict(model, grid.copy(), 1914, selected_model)
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # --- Dashboard ---
        st.markdown("### 📊 Executive Summary")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        plan = calculate_inventory_plan(forecast_grid, 1914, rmse, lead_time, z_score, current_stock)
        total_demand = plan['sales'].sum()
        next_order = plan[plan['recommended_order'] > 0]
        
        if not next_order.empty:
            if 'date' in next_order.columns:
                next_order_day = next_order['date'].dt.strftime('%Y-%m-%d').iloc[0]
            else:
                next_order_day = f"Day {next_order['d_num'].iloc[0]}"
        else:
            next_order_day = "None"
        
        kpi1.metric("Model Confidence (RMSE)", f"{rmse:.2f}", delta_color="inverse")
        kpi2.metric("28-Day Demand Forecast", f"{int(total_demand)} units")
        kpi3.metric("Safety Stock Buffer", f"{int(plan['safety_stock'].mean())} units")
        kpi4.metric("Next Replenishment", next_order_day)

        # --- Tabs ---
        tab1, tab2, tab3 = st.tabs(["📈 Lifecycle & Forecast", "🚚 Inventory Action Plan", "🔍 Model Diagnostics"])
        
        with tab1:
            st.plotly_chart(plot_interactive_lifecycle(grid, plan, selected_item, rmse), use_container_width=True)
            
        with tab2:
            st.plotly_chart(plot_interactive_action_plan(plan, selected_item), use_container_width=True)
            st.markdown("#### 📋 Detailed Replenishment Schedule")
            
            cols_to_show = ['sales', 'safety_stock', 'projected_inventory', 'recommended_order']
            if 'date' in plan.columns:
                display_df = plan[['date'] + cols_to_show].head(14)
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            else:
                display_df = plan[['d_num'] + cols_to_show].head(14)

            st.dataframe(
                display_df.style.background_gradient(cmap='Reds', subset=['recommended_order'])
                                .format({c: "{:.1f}" for c in cols_to_show}),
                use_container_width=True
            )
            
        with tab3:
            if selected_model == "Prophet":
                st.info("Prophet operates on Bayesian curve fitting (Trend + Seasonality). Feature Importance is not applicable like Tree models.")
                st.markdown("**Prophet Components:** The model relies heavily on the `sell_price` regressor and weekly seasonality patterns found in this dataset.")
            else:
                st.plotly_chart(plot_feature_importance(model, final_features, selected_model), use_container_width=True)
                st.info("Feature Importance shows which variables (Lags, Trends, Price) most influenced the AI's decision.")

if __name__ == "__main__":
    main()