import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings

# ==========================================
# 0. High-Performance Configuration
# ==========================================
st.set_page_config(
    page_title="Supply Chain AI Cortex | Global Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)
warnings.filterwarnings('ignore')

st.markdown("""
<style>
    .stMetric {
        background-color: #0E1117;
        border: 1px solid #30333F;
        padding: 15px;
        border-radius: 5px;
    }
    .stProgress .st-bo { background-color: #00AA00; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Global Data Pipeline & Extrapolation
# ==========================================
@st.cache_data
def load_lean_data(filepath='m5_lean.csv'):
    try:
        df = pd.read_csv(filepath)
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

def extend_grid_global(df, horizon, start_day):
    df = df.copy()
    max_day = df['d_num'].max()
    target_end_day = start_day + horizon - 1
    
    if target_end_day > max_day:
        future_rows = []
        last_states = df[df['d_num'] == max_day]
        
        for d in range(max_day + 1, target_end_day + 1):
            for _, row in last_states.iterrows():
                new_row = row.copy()
                new_row['d'] = f"d_{d}"
                new_row['d_num'] = d
                new_row['date'] = row['date'] + pd.Timedelta(days=d - max_day)
                new_row['sales'] = np.nan 
                future_rows.append(new_row)
        
        df = pd.concat([df, pd.DataFrame(future_rows)], ignore_index=True)
    return df

def create_features_global(df):
    """
    ADVANCED RIGOR: Added dayofmonth to capture payday cyclicality.
    """
    df = df.copy().sort_values(by=['item_id', 'd_num'])
    
    # 1. Time-Series Cyclicality
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    df['dayofmonth'] = df['date'].dt.day # Critical for payday dynamics

    # 2. Autoregressive Lags (Grouped by Item)
    df['lag_1'] = df.groupby('item_id')['sales'].shift(1)
    df['lag_7'] = df.groupby('item_id')['sales'].shift(7)
    df['lag_14'] = df.groupby('item_id')['sales'].shift(14)
    df['lag_28'] = df.groupby('item_id')['sales'].shift(28)
    
    # 3. Rolling Statistical Windows
    df['shifted_sales'] = df['lag_1']
    df['rolling_mean_7'] = df.groupby('item_id')['shifted_sales'].transform(lambda x: x.rolling(7).mean())
    df['rolling_std_7'] = df.groupby('item_id')['shifted_sales'].transform(lambda x: x.rolling(7).std())
    df['rolling_mean_28'] = df.groupby('item_id')['shifted_sales'].transform(lambda x: x.rolling(28).mean())
    
    # 4. Price Elasticity Signals
    df['price_change'] = df.groupby('item_id')['sell_price'].pct_change()
    df['price_volatility'] = df.groupby('item_id')['sell_price'].transform(lambda x: x.rolling(7).std())
    
    df.drop(columns=['shifted_sales'], inplace=True)
    return df

def prepare_global_training_data(df, selected_items, store_id):
    data = df[(df['item_id'].isin(selected_items)) & (df['store_id'] == store_id)].copy()
    
    le_event = LabelEncoder()
    data['event_name_1'] = data['event_name_1'].astype(str).fillna('None')
    data['event_name_1'] = le_event.fit_transform(data['event_name_1']).astype(int) 
    
    le_item = LabelEncoder()
    data['item_id_encoded'] = le_item.fit_transform(data['item_id']).astype(int)
    
    # Required for XGBoost categorical support
    data['item_id_encoded'] = data['item_id_encoded'].astype('category')
    data['event_name_1'] = data['event_name_1'].astype('category')
    
    return data, le_item

# ==========================================
# 2. Global Model Architecture
# ==========================================
def train_lightgbm_global(X_train, y_train, X_val, y_val, lr, iterations):
    params = {
        'objective': 'tweedie', 'tweedie_variance_power': 1.1, 'metric': 'rmse',
        'learning_rate': lr, 'num_leaves': 128, 'max_depth': 8,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 
        'lambda_l1': 0.5, 'lambda_l2': 0.5, 'n_jobs': -1, 'verbosity': -1
    }
    categorical_features = ['item_id_encoded', 'event_name_1', 'dayofweek', 'is_weekend', 'month', 'dayofmonth']
    valid_cats = [c for c in categorical_features if c in X_train.columns]
    
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=valid_cats)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, categorical_feature=valid_cats)
    return lgb.train(params, train_set, num_boost_round=iterations, valid_sets=[train_set, val_set], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

def train_xgboost_global(X_train, y_train, X_val, y_val, lr, iterations):
    """Added Global XGBoost Engine with enable_categorical=True"""
    y_train = y_train.astype(float)
    y_val = y_val.astype(float)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    
    params = {
        'objective': 'reg:tweedie', 'tweedie_variance_power': 1.1, 'eval_metric': 'rmse', 
        'eta': lr, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'alpha': 0.5, 'lambda': 0.5, 'nthread': -1, 'verbosity': 0
    }
    return xgb.train(params, dtrain, num_boost_round=iterations, evals=[(dtrain, 'train'), (dval, 'eval')], early_stopping_rounds=100, verbose_eval=False)

def recursive_predict_global(model, df, start_day, horizon, model_type):
    exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'sales', 'd', 'date', 'wm_yr_wk', 'd_num']
    
    progress_text = "Running Vectorized Global Simulation..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, day in enumerate(range(start_day, start_day + horizon)):
        df = create_features_global(df)
        
        # Must maintain category types for XGBoost during prediction
        if 'item_id_encoded' in df.columns: df['item_id_encoded'] = df['item_id_encoded'].astype('category')
        if 'event_name_1' in df.columns: df['event_name_1'] = df['event_name_1'].astype('category')
            
        features = [c for c in df.columns if c not in exclude_cols]
        
        mask = df['d_num'] == day
        X_test = df[mask][features]
        if X_test.empty: break
        
        if model_type == 'LightGBM': 
            preds = model.predict(X_test)
        elif model_type == 'XGBoost':
            preds = model.predict(xgb.DMatrix(X_test, enable_categorical=True))
        elif model_type == 'CatBoost': 
            preds = model.predict(X_test)
            
        df.loc[mask, 'sales'] = np.maximum(0, preds) 
        my_bar.progress((i + 1) / horizon, text=f"Simulating Day {day} across Global Tensor...")
        
    my_bar.empty()
    return df, features

def calculate_inventory_plan(df, start_day, horizon, rmse, lead_time, service_level_z, initial_stock, item_id):
    plan_mask = (df['d_num'] >= start_day) & (df['d_num'] < start_day + horizon) & (df['item_id'] == item_id)
    plan = df[plan_mask].copy().sort_values(by='d_num')
    plan['safety_stock'] = service_level_z * rmse * np.sqrt(lead_time)
    
    inventory_levels, order_quantities = [], []
    stock = initial_stock
    
    for _, row in plan.iterrows():
        reorder_point = row['safety_stock'] + (row['sales'] * lead_time)
        order_qty = 0
        if stock < reorder_point:
            order_qty = (reorder_point + (row['sales'] * 7)) - stock
            stock += order_qty 
        stock -= row['sales']
        inventory_levels.append(max(0, stock))
        order_quantities.append(max(0, order_qty))
        
    plan['projected_inventory'] = inventory_levels
    plan['recommended_order'] = order_quantities
    return plan

# ==========================================
# 3. Interactive Visualization
# ==========================================
def plot_portfolio_comparison(portfolio_details):
    fig = go.Figure()
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
    
    for idx, (item, details) in enumerate(portfolio_details.items()):
        plan = details['plan']
        x_axis = plan['date'] if 'date' in plan.columns else plan['d_num']
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(x=x_axis, y=plan['projected_inventory'], name=f"{item}", mode='lines', line=dict(width=2, color=color)))
        
    fig.update_layout(title=dict(text="<b>🌐 Global Portfolio Convergence</b>", font=dict(size=18)), template="plotly_white", hovermode="x unified", height=450, yaxis_title="Units on Hand", xaxis_title="Timeline")
    return fig

def plot_interactive_lifecycle(history, forecast, ground_truth, item, rmse):
    """
    VISUAL PERFECTION: Seamless single gray line for all actual data (history + truth).
    Green forecast overlaid exactly on top.
    """
    history_view = history[history['d_num'] > (history['d_num'].max() - 120)]
    
    # 1. The Seamless Reality Line
    reality = pd.concat([history_view, ground_truth]).dropna(subset=['sales']).sort_values(by='d_num')
    
    fig = go.Figure()
    
    x_real = reality['date'] if 'date' in reality.columns else reality['d_num']
    fig.add_trace(go.Scatter(
        x=x_real, y=reality['sales'], mode='lines', name='Actual Sales (Reality)',
        line=dict(color='gray', width=1.5), opacity=0.7
    ))
    
    # 2. The AI Foresight
    x_fore = forecast['date'] if 'date' in forecast.columns else forecast['d_num']
    fig.add_trace(go.Scatter(
        x=x_fore, y=forecast['sales'], mode='lines+markers', name='AI Forecast',
        line=dict(color='#00CC96', width=3), marker=dict(size=5)
    ))
    
    # 3. The Probability Bounds
    upper_bound = forecast['sales'] + (1.96 * rmse)
    lower_bound = forecast['sales'].apply(lambda x: max(0, x - (1.96 * rmse)))
    
    fig.add_trace(go.Scatter(
        x=pd.concat([x_fore, x_fore[::-1]]),
        y=pd.concat([upper_bound, lower_bound[::-1]]),
        fill='toself', fillcolor='rgba(0, 204, 150, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title=f"<b>Deep Learning Trajectory vs Reality: {item}</b>", xaxis_title="Date", yaxis_title="Sales Volume",
        template="plotly_white", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450
    )
    return fig

def plot_interactive_action_plan(plan, item):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x_axis = plan['date'] if 'date' in plan.columns else plan['d_num']
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=plan['safety_stock'], name="Safety Stock (Risk Threshold)",
        line=dict(color='rgba(255, 99, 71, 0.5)', width=1, dash='dash'), fill='tozeroy',
        fillcolor='rgba(255, 99, 71, 0.1)', hoverinfo='skip'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x_axis, y=plan['sales'], name="Predicted Daily Demand",
        line=dict(color='#A0A0A0', width=1.5, dash='dot'), opacity=0.6
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=plan['projected_inventory'], name="Projected Stock Level",
        line=dict(color='#1F77B4', width=3), fill='tonexty', fillcolor='rgba(31, 119, 180, 0.05)'
    ), secondary_y=False)
    
    orders = plan[plan['recommended_order'] > 0]
    if not orders.empty:
        x_orders = orders['date'] if 'date' in orders.columns else orders['d_num']
        fig.add_trace(go.Bar(
            x=x_orders, y=orders['recommended_order'], name="Replenishment Order (Qty)",
            marker_color='#2CA02C', opacity=0.9, text=orders['recommended_order'].astype(int), textposition='auto'
        ), secondary_y=True)

    fig.update_layout(
        title=dict(text=f"<b>🛡️ Inventory Strategy & Replenishment Plan: {item}</b>", font=dict(size=20)),
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.5)"),
        height=550 
    )
    
    fig.update_yaxes(title_text="<b>Stock Level (Units)</b>", secondary_y=False, showgrid=True, gridcolor='#F0F0F0')
    max_order = plan['recommended_order'].max() if not plan['recommended_order'].empty else 10
    fig.update_yaxes(title_text="<b>Order Quantity</b>", secondary_y=True, showgrid=False, range=[0, max_order * 1.5])
    return fig

# ==========================================
# 4. Main Application Logic
# ==========================================
def main():
    st.title("Supply Chain AI Cortex | Global Edition")
    st.markdown("### Unified Omniscient Architecture for Category Processing")
    
    data = load_lean_data()
    if data is None:
        st.error("⚠️ System Offline: `m5_lean.csv` not detected.")
        return

    with st.sidebar:
        st.header("⚙️ System Configuration")
        selected_store = st.selectbox("Store Location", data['store_id'].unique())
        
        available_items = data['item_id'].unique()
        selected_items = st.multiselect("Select Product Portfolio", available_items, default=available_items[:3])
        
        st.subheader("Global Parameters")
        forecast_horizon = st.slider("Forecast Horizon (Days)", 7, 90, 28, 7)
        lead_time = st.slider("Supplier Lead Time (Days)", 1, 14, 3)
        service_level = st.slider("Target Service Level (%)", 80, 99, 95)
        current_stock = st.number_input("Current Stock Per Item", min_value=0, value=50)
        z_score = {99: 2.33, 95: 1.64, 90: 1.28, 85: 1.04, 80: 0.84}.get(service_level, 1.64)

        with st.expander("🛠️ Advanced AI Parameters"):
            # RESTORED: XGBoost is back as a Global contender
            selected_model = st.radio("Architecture", ["LightGBM", "XGBoost", "CatBoost"], index=0)
            learning_rate = st.number_input("Learning Rate", value=0.03)
            iterations = st.number_input("Boosting Iterations", value=1000, step=100)

    if st.button("🚀 Run Global AI Engine", type="primary"):
        if not selected_items:
            st.warning("Please select at least one item.")
            return

        SPLIT_DAY = 1913
        st.session_state['portfolio_details'] = {} 
        master_metrics = {'total_demand': 0, 'capital_tied': 0, 'item_rmses': {}}
        
        with st.status("Initializing Unified Global Engine...", expanded=True) as status:
            st.write("1. 🔧 Building Multi-Dimensional Tensor...")
            grid, le_item = prepare_global_training_data(data, selected_items, selected_store)
            grid = extend_grid_global(grid, forecast_horizon, SPLIT_DAY + 1)
            
            st.write("2. 🧮 Engineering Temporal & Cyclical Features...")
            grid_feat = create_features_global(grid)
            grid_feat_train = grid_feat.dropna(subset=['lag_28', 'rolling_mean_28', 'sales'])
            
            exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'sales', 'd', 'date', 'wm_yr_wk', 'd_num']
            features = [c for c in grid_feat_train.columns if c not in exclude_cols]
            
            train_mask = grid_feat_train['d_num'] <= SPLIT_DAY - 28
            val_mask = (grid_feat_train['d_num'] > SPLIT_DAY - 28) & (grid_feat_train['d_num'] <= SPLIT_DAY)
            
            X_train, y_train = grid_feat_train[train_mask][features], grid_feat_train[train_mask]['sales']
            X_val, y_val = grid_feat_train[val_mask][features], grid_feat_train[val_mask]['sales']
            
            st.write(f"3. 🧠 Training Single Global Model ({selected_model})...")
            if selected_model == "LightGBM":
                model = train_lightgbm_global(X_train, y_train, X_val, y_val, learning_rate, iterations)
                val_preds = model.predict(X_val)
            elif selected_model == "XGBoost":
                model = train_xgboost_global(X_train, y_train, X_val, y_val, learning_rate, iterations)
                val_preds = model.predict(xgb.DMatrix(X_val, enable_categorical=True))
            elif selected_model == "CatBoost":
                cat_features = ['item_id_encoded', 'event_name_1', 'dayofweek', 'is_weekend', 'month', 'dayofmonth']
                valid_cats = [c for c in cat_features if c in features]
                cat_indices = [features.index(c) for c in valid_cats]
                
                train_pool = Pool(X_train, y_train, cat_features=cat_indices)
                val_pool = Pool(X_val, y_val, cat_features=cat_indices)
                
                model = CatBoostRegressor(
                    iterations=iterations, learning_rate=learning_rate, depth=6, 
                    loss_function='Tweedie:variance_power=1.1', verbose=0, l2_leaf_reg=5 
                )
                model.fit(train_pool, eval_set=val_pool)
                val_preds = model.predict(X_val)

            X_val_df = grid_feat_train[val_mask].copy()
            X_val_df['val_preds'] = val_preds
            for item in selected_items:
                item_val = X_val_df[X_val_df['item_id'] == item]
                rmse = np.sqrt(mean_squared_error(item_val['sales'], item_val['val_preds']))
                master_metrics['item_rmses'][item] = rmse

            st.write("4. 🔮 Executing Vectorized Future Simulation...")
            forecast_grid, _ = recursive_predict_global(model, grid.copy(), SPLIT_DAY + 1, forecast_horizon, selected_model)
            
            st.write("5. 🚚 Extrapolating Inventory Plans...")
            portfolio_plans = []
            for item in selected_items:
                rmse = master_metrics['item_rmses'][item]
                item_plan = calculate_inventory_plan(forecast_grid, SPLIT_DAY + 1, forecast_horizon, rmse, lead_time, z_score, current_stock, item)
                portfolio_plans.append(item_plan)
                
                st.session_state['portfolio_details'][item] = {
                    'history_grid': data[(data['d_num'] <= SPLIT_DAY) & (data['item_id'] == item)],
                    'ground_truth': data[(data['d_num'] > SPLIT_DAY) & (data['d_num'] <= SPLIT_DAY + forecast_horizon) & (data['item_id'] == item)],
                    'plan': item_plan,
                    'rmse': rmse
                }
                
                avg_price = item_plan['sell_price'].mean() if 'sell_price' in item_plan.columns else 0
                master_metrics['total_demand'] += item_plan['sales'].sum()
                master_metrics['capital_tied'] += (item_plan['safety_stock'].mean() * avg_price)

            st.session_state['master_df'] = pd.concat(portfolio_plans, ignore_index=True)
            st.session_state['avg_rmse'] = np.mean(list(master_metrics['item_rmses'].values()))
            st.session_state['total_demand'] = master_metrics['total_demand']
            st.session_state['capital_tied'] = master_metrics['capital_tied']
            
            status.update(label=f"✅ Unified AI Processed {len(selected_items)} Nodes Matrix in O(1) loop logic.", state="complete", expanded=False)

    if 'portfolio_details' in st.session_state and st.session_state['portfolio_details']:
        st.markdown("### 🌐 Global Category Executive Summary")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Average Category RMSE", f"{st.session_state['avg_rmse']:.2f}")
        kpi2.metric("Total Category Demand", f"{int(st.session_state['total_demand'])} units")
        kpi3.metric("Total Capital Tied in Safety Stock", f"${int(st.session_state['capital_tied'])}")

        st.markdown("---")
        tab1, tab2 = st.tabs(["📊 Unified Analytics Console", "🗂️ Master Schedule Export"])
        
        with tab1:
            st.plotly_chart(plot_portfolio_comparison(st.session_state['portfolio_details']), use_container_width=True)
            st.markdown("---")
            st.markdown("#### 🔬 Dynamic SKU Inspection")
            inspect_item = st.selectbox("Select specific node for granular analysis:", list(st.session_state['portfolio_details'].keys()))
            
            if inspect_item:
                details = st.session_state['portfolio_details'][inspect_item]
                
                st.plotly_chart(
                    plot_interactive_lifecycle(details['history_grid'], details['plan'], details['ground_truth'], inspect_item, details['rmse']), 
                    use_container_width=True
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.plotly_chart(
                    plot_interactive_action_plan(details['plan'], inspect_item), 
                    use_container_width=True
                )
        with tab2:
            st.markdown("#### 💾 Master Replenishment Plan")
            export_df = st.session_state['master_df'][['date', 'item_id', 'sales', 'safety_stock', 'projected_inventory', 'recommended_order']].copy()
            export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
            actionable_orders = export_df[export_df['recommended_order'] > 0].sort_values(by=['date', 'item_id'])
            
            st.download_button("📥 Download Global ERP Matrix (CSV)", data=export_df.to_csv(index=False).encode('utf-8'), file_name='global_portfolio_plan.csv', mime='text/csv', type="primary")
            st.dataframe(actionable_orders.style.background_gradient(cmap='Greens', subset=['recommended_order']).format(precision=1), use_container_width=True)

if __name__ == "__main__":
    main()