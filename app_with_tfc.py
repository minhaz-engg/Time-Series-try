import streamlit as st
import pandas as pd
import numpy as np
import math
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from prophet import Prophet 
import warnings
import logging

# --- PyTorch Imports for TF-C ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom CSS for "Dark Mode" Professional Look & Terminal
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
    .log-terminal {
        background-color: #000000;
        color: #00FF00;
        font-family: monospace;
        font-size: 12px;
        padding: 10px;
        border-radius: 5px;
        height: 200px;
        overflow-y: scroll;
        margin-bottom: 15px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Data Pipeline
# ==========================================
@st.cache_data
def load_lean_data(filepath='mini_data/mini_sales.csv', cal_path='mini_data/mini_calendar.csv', price_path='mini_data/mini_prices.csv'):
    """Loads and caches the optimized dataset. Adjust paths to match your Codespace."""
    try:
        # If you are using the single m5_lean.csv, modify this block. 
        # Assuming the merged m5_lean.csv structure for this unified code:
        df = pd.read_csv('m5_lean.csv')
        
        # Robust Date Reconstruction
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
    """Rigorously regenerates temporal features for Tree Models."""
    df = df.copy()
    
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['lag_14'] = df['sales'].shift(14)
    df['lag_28'] = df['sales'].shift(28)
    
    df['rolling_mean_7'] = df['sales'].shift(1).rolling(7).mean()
    df['rolling_std_7'] = df['sales'].shift(1).rolling(7).std()
    df['rolling_mean_28'] = df['sales'].shift(1).rolling(28).mean()
    
    df['price_change'] = df['sell_price'].pct_change()
    df['price_volatility'] = df['sell_price'].rolling(7).std()
    
    return df

def prepare_training_data(df, item_id, store_id):
    """Filters and encodes data for a specific time-series."""
    data = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
    
    data['wday'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month - 1
    
    le = LabelEncoder()
    data['event_name_1'] = data['event_name_1'].astype(str).fillna('None')
    data['event_name_1'] = le.fit_transform(data['event_name_1'])
    data['event_name_1'] = data['event_name_1'].astype(int) 
    
    for col in ['snap_CA', 'snap_TX', 'snap_WI']:
        if col in data.columns:
            data[col] = data[col].fillna(0).astype(int)
            
    return data

# ==========================================
# 2. PyTorch TF-C Architecture (Corrected)
# ==========================================
class PositionalEncoding(nn.Module):
    """Injects absolute temporal positioning into the sequence."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class AdvancedTFC(nn.Module):
    def __init__(self, seq_len, freq_len, num_cont_features, cat_cardinalities, d_model=64):
        super(AdvancedTFC, self).__init__()
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(10, (card + 1) // 2)) for card in cat_cardinalities
        ])
        self.cat_dim = sum([e.embedding_dim for e in self.embeddings])
        
        self.time_input_proj = nn.Linear(num_cont_features + self.cat_dim, d_model)
        self.pos_encoder_t = PositionalEncoding(d_model=d_model, max_len=seq_len)
        
        encoder_layer_t = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*2, 
            batch_first=True, dropout=0.2, norm_first=True
        )
        self.time_encoder = nn.TransformerEncoder(encoder_layer_t, num_layers=2)
        
        self.projector_t = nn.Sequential(
            nn.Linear(d_model * seq_len, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

        self.freq_input_proj = nn.Linear(1, d_model)
        encoder_layer_f = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*2, 
            batch_first=True, dropout=0.2, norm_first=True
        )
        self.freq_encoder = nn.TransformerEncoder(encoder_layer_f, num_layers=2)
        
        self.projector_f = nn.Sequential(
            nn.Linear(d_model * freq_len, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

    def forward(self, x_cont, x_cat, x_freq):
        embeddings = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x_cat_embed = torch.cat(embeddings, dim=-1)
        x_time_combined = torch.cat([x_cont, x_cat_embed], dim=-1)
        
        h_time = self.time_input_proj(x_time_combined)
        h_time = self.pos_encoder_t(h_time) 
        h_time = self.time_encoder(h_time)
        h_time_flat = h_time.reshape(h_time.shape[0], -1)
        z_time = self.projector_t(h_time_flat)

        h_freq = self.freq_input_proj(x_freq)
        h_freq = self.freq_encoder(h_freq)
        h_freq_flat = h_freq.reshape(h_freq.shape[0], -1)
        z_freq = self.projector_f(h_freq_flat)

        return h_time_flat, z_time, h_freq_flat, z_freq

class ForecasterHead(nn.Module):
    def __init__(self, time_dim, freq_dim):
        super(ForecasterHead, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(time_dim + freq_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, h_t_flat, h_f_flat):
        return self.regressor(torch.cat((h_t_flat, h_f_flat), dim=1))

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        batch_size = z_i.shape[0]
        
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        mask = torch.eye(2 * batch_size, device=DEVICE).bool()
        similarity_matrix.masked_fill_(mask, -9e9)
        
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=DEVICE),
            torch.arange(0, batch_size, device=DEVICE)
        ], dim=0)
        
        loss = self.criterion(similarity_matrix, labels)
        return loss / (2 * batch_size)

# ==========================================
# 3. Training & Simulation Engines
# ==========================================

# --- TF-C Engine ---
def prepare_tfc_tensors(df, seq_len=60):
    sales_scaler = StandardScaler()
    price_scaler = StandardScaler()
    
    work_df = df.copy()
    work_df['sales'] = work_df['sales'].fillna(0)
    work_df['sell_price'] = work_df['sell_price'].fillna(method='ffill').fillna(0)
    
    work_df['sales_scaled'] = sales_scaler.fit_transform(work_df[['sales']])
    work_df['price_scaled'] = price_scaler.fit_transform(work_df[['sell_price']])
    
    data_cont = work_df[['sales_scaled', 'price_scaled']].values
    
    cat_cols = ['wday', 'month', 'event_name_1']
    for snap in ['snap_CA', 'snap_TX', 'snap_WI']:
        if snap in work_df.columns: cat_cols.append(snap)
    data_cat = work_df[cat_cols].values.astype(int)
    
    X_cont, X_cat, X_freq, y = [], [], [], []
    
    for i in range(len(work_df) - seq_len):
        X_cont.append(data_cont[i : i + seq_len])
        X_cat.append(data_cat[i : i + seq_len])
        
        sales_window = data_cont[i : i + seq_len, 0]
        fft_mag = np.abs(np.fft.rfft(sales_window))
        fft_mag = (fft_mag - np.mean(fft_mag)) / (np.std(fft_mag) + 1e-6)
        X_freq.append(fft_mag)
        
        y.append(data_cont[i + seq_len, 0])
        
    X_cont_t = torch.FloatTensor(np.array(X_cont))
    X_cat_t = torch.LongTensor(np.array(X_cat))
    X_freq_t = torch.FloatTensor(np.array(X_freq)).unsqueeze(-1)
    y_t = torch.FloatTensor(np.array(y)).unsqueeze(-1)
    
    scalers = {'sales': sales_scaler, 'price': price_scaler}
    cat_cards = [work_df[col].max() + 1 for col in cat_cols]
    
    return X_cont_t, X_cat_t, X_freq_t, y_t, scalers, cat_cards, cat_cols

def train_tfc_model(X_cont, X_cat, X_freq, y, cat_cards, log_placeholder, pbar, epochs_pre=10, epochs_fine=15):
    dataset = TensorDataset(X_cont, X_cat, X_freq, y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    model = AdvancedTFC(seq_len=X_cont.shape[1], freq_len=X_freq.shape[1], 
                        num_cont_features=X_cont.shape[2], cat_cardinalities=cat_cards, d_model=64).to(DEVICE)
    
    head = ForecasterHead(64 * X_cont.shape[1], 64 * X_freq.shape[1]).to(DEVICE)
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-5)
    
    log_text = "<b>[SYSTEM] Initializing TF-C Transformer Protocol...</b><br>"
    log_placeholder.markdown(f"<div class='log-terminal'>{log_text}</div>", unsafe_allow_html=True)
    
    total_epochs = epochs_pre + epochs_fine
    current_epoch = 0

    nt_loss = NTXentLoss().to(DEVICE)
    for epoch in range(epochs_pre):
        model.train()
        total_loss = 0
        for batch in train_loader:
            xc, xcat, xf, _ = [b.to(DEVICE) for b in batch]
            xc_aug = xc + (torch.randn_like(xc) * 0.05)
            xf_aug = xf + (torch.randn_like(xf) * 0.05)
            
            optimizer.zero_grad()
            _, z_t, _, z_f = model(xc, xcat, xf)
            _, z_t_aug, _, z_f_aug = model(xc_aug, xcat, xf_aug)
            
            loss = 0.2 * (nt_loss(z_t, z_t_aug) + nt_loss(z_f, z_f_aug)) + nt_loss(z_t, z_f)
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            total_loss += loss.item()
            
        current_epoch += 1
        pbar.progress(current_epoch / total_epochs, text="Phase 1: Contrastive Pre-Training...")
        log_text = f"[Phase 1] Epoch {epoch+1}/{epochs_pre} | Contrastive Loss: {total_loss/len(train_loader):.4f}<br>" + log_text
        log_placeholder.markdown(f"<div class='log-terminal'>{log_text}</div>", unsafe_allow_html=True)

    criterion = nn.MSELoss()
    final_val_mse = 0
    for epoch in range(epochs_fine):
        model.train()
        head.train()
        total_mse = 0
        for batch in train_loader:
            xc, xcat, xf, yt = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            h_t, _, h_f, _ = model(xc, xcat, xf)
            preds = head(h_t, h_f)
            loss = criterion(preds, yt)
            
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 4.0)
            optimizer.step()
            total_mse += loss.item()
            
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                xc, xcat, xf, yt = [b.to(DEVICE) for b in batch]
                h_t, _, h_f, _ = model(xc, xcat, xf)
                val_mse += criterion(head(h_t, h_f), yt).item()
                
        final_val_mse = val_mse/len(val_loader) if len(val_loader) > 0 else 0.01
        current_epoch += 1
        pbar.progress(current_epoch / total_epochs, text="Phase 2: Supervised Fine-Tuning...")
        log_text = f"[Phase 2] Epoch {epoch+1}/{epochs_fine} | Train MSE: {total_mse/len(train_loader):.4f} | Val MSE: {final_val_mse:.4f}<br>" + log_text
        log_placeholder.markdown(f"<div class='log-terminal'>{log_text}</div>", unsafe_allow_html=True)

    return model, head, final_val_mse

def recursive_predict_tfc(model, head, grid, scalers, cat_cols, start_day, seq_len=60):
    model.eval()
    head.eval()
    
    future_horizon = 28
    train_history = grid[grid['d_num'] < start_day].iloc[-seq_len:].copy()
    
    future_mask = grid['d_num'] >= start_day
    future_meta = grid[future_mask].iloc[:future_horizon].copy()
    
    current_window = train_history.copy()
    predictions = []
    
    for i in range(future_horizon):
        s_data = scalers['sales'].transform(current_window[['sales']].fillna(0)).reshape(-1,1)
        p_data = scalers['price'].transform(current_window[['sell_price']].fillna(0)).reshape(-1,1)
        xc = torch.FloatTensor(np.hstack([s_data, p_data])).unsqueeze(0).to(DEVICE)
        
        cats = current_window[cat_cols].values.astype(int)
        xcat = torch.LongTensor(cats).unsqueeze(0).to(DEVICE)
        
        fft = np.abs(np.fft.rfft(s_data.flatten()))
        xf = torch.FloatTensor((fft - np.mean(fft))/(np.std(fft)+1e-6)).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        
        with torch.no_grad():
            ht, _, hf, _ = model(xc, xcat, xf)
            pred_scaled = head(ht, hf).item()
            
        pred_real = scalers['sales'].inverse_transform([[pred_scaled]])[0][0]
        pred_real = max(0, pred_real)
        predictions.append(pred_real)
        
        next_row = future_meta.iloc[i].copy()
        next_row['sales'] = pred_real
        current_window = pd.concat([current_window.iloc[1:], next_row.to_frame().T])

    grid.loc[future_mask, 'sales'] = predictions
    return grid

# --- Tree and Prophet Functions ---
def train_lightgbm(X_train, y_train, X_val, y_val):
    params = {'objective': 'tweedie', 'tweedie_variance_power': 1.1, 'metric': 'rmse',
              'learning_rate': 0.03, 'num_leaves': 64, 'feature_fraction': 0.8,
              'bagging_fraction': 0.7, 'bagging_freq': 1, 'lambda_l1': 0.1, 
              'lambda_l2': 0.1, 'n_jobs': -1, 'verbosity': -1}
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    return lgb.train(params, train_set, num_boost_round=1500, valid_sets=[train_set, val_set],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

def train_xgboost(X_train, y_train, X_val, y_val):
    y_train, y_val = y_train.astype(float), y_val.astype(float)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    params = {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.1, 'eval_metric': 'rmse',
              'eta': 0.03, 'max_depth': 8, 'subsample': 0.7, 'colsample_bytree': 0.7,
              'alpha': 0.1, 'lambda': 0.1, 'nthread': -1, 'verbosity': 0}
    return xgb.train(params, dtrain, num_boost_round=1500, evals=[(dtrain, 'train'), (dval, 'eval')],
                     early_stopping_rounds=100, verbose_eval=False)

def train_catboost(X_train, y_train, X_val, y_val, features):
    cat_cols = ['event_name_1', 'snap_CA', 'snap_TX', 'snap_WI']
    cat_indices = [features.index(c) for c in cat_cols if c in features]
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    model = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                              loss_function='Tweedie:variance_power=1.1', eval_metric='RMSE',
                              random_seed=42, early_stopping_rounds=100, verbose=0, allow_writing_files=False)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model

def train_prophet_model(df_train):
    prophet_df = df_train[['date', 'sales', 'sell_price']].copy()
    prophet_df.columns = ['ds', 'y', 'sell_price']
    model = Prophet(growth='linear', daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, seasonality_mode='multiplicative', interval_width=0.95)
    model.add_regressor('sell_price')
    model.fit(prophet_df)
    return model

def recursive_predict(model, df, start_day, model_type):
    df = create_features(df)
    exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                    'sales', 'd', 'date', 'wm_yr_wk', 'd_num']
    features = [c for c in df.columns if c not in exclude_cols]
    
    pbar = st.progress(0, text="Running Recursive Stochastic Simulation...")
    for i, day in enumerate(range(start_day, start_day + 28)):
        df = create_features(df)
        mask = df['d_num'] == day
        X_test = df[mask][features]
        if X_test.empty: break
            
        if model_type == 'LightGBM' or model_type == 'CatBoost':
            pred = model.predict(X_test)[0]
        elif model_type == 'XGBoost':
            dtest = xgb.DMatrix(X_test, enable_categorical=True)
            pred = model.predict(dtest)[0]
            
        df.loc[mask, 'sales'] = max(0, pred)
        pbar.progress((i + 1) / 28, text=f"Simulating Day {day}...")
    pbar.empty()
    return df, features

def batch_predict_prophet(model, grid, start_day):
    future_mask = grid['d_num'] >= start_day
    future_df = grid[future_mask][['date', 'sell_price']].copy()
    future_df.columns = ['ds', 'sell_price']
    forecast = model.predict(future_df)
    grid.loc[future_mask, 'sales'] = forecast['yhat'].values
    return grid, ['sell_price', 'trend', 'weekly', 'yearly']

# ==========================================
# 4. Logistics & Visualization
# ==========================================
def calculate_inventory_plan(df, start_day, rmse, lead_time, service_level_z, initial_stock):
    plan = df[df['d_num'] >= start_day].copy()
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

def plot_interactive_lifecycle(history, forecast, item, rmse):
    history_view = history[history['d_num'] > (1913 - 120)]
    fig = go.Figure()
    x_hist = history_view['date'] if 'date' in history_view.columns else history_view['d_num']
    x_fore = forecast['date'] if 'date' in forecast.columns else forecast['d_num']
    
    fig.add_trace(go.Scatter(x=x_hist, y=history_view['sales'], mode='lines', name='Historical Sales', line=dict(color='gray', width=1.5), opacity=0.6))
    fig.add_trace(go.Scatter(x=x_fore, y=forecast['sales'], mode='lines+markers', name='AI Forecast', line=dict(color='#00CC96', width=3), marker=dict(size=5)))
    
    upper = forecast['sales'] + (1.96 * rmse)
    lower = forecast['sales'] - (1.96 * rmse)
    lower = lower.apply(lambda x: max(0, x))
    fig.add_trace(go.Scatter(x=pd.concat([x_fore, x_fore[::-1]]), y=pd.concat([upper, lower[::-1]]), fill='toself', fillcolor='rgba(0, 204, 150, 0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='95% Confidence Interval'))
    
    fig.update_layout(title=f"<b>Deep Learning Trajectory: {item}</b>", xaxis_title="Date", yaxis_title="Sales Volume", template="plotly_white", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

def plot_interactive_action_plan(plan, item):
    """Advanced Inventory Action Plan Plot with Risk Zones and Replenishment Bars"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x_axis = plan['date'] if 'date' in plan.columns else plan['d_num']
    
    fig.add_trace(go.Scatter(x=x_axis, y=plan['safety_stock'], name="Safety Stock (Risk Threshold)", line=dict(color='rgba(255, 99, 71, 0.5)', width=1, dash='dash'), fill='tozeroy', fillcolor='rgba(255, 99, 71, 0.1)', hoverinfo='skip'), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=plan['sales'], name="Predicted Daily Demand", line=dict(color='#A0A0A0', width=1.5, dash='dot'), opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=plan['projected_inventory'], name="Projected Stock Level", line=dict(color='#1F77B4', width=3), fill='tonexty', fillcolor='rgba(31, 119, 180, 0.05)'), secondary_y=False)
    
    orders = plan[plan['recommended_order'] > 0]
    if not orders.empty:
        x_orders = orders['date'] if 'date' in orders.columns else orders['d_num']
        fig.add_trace(go.Bar(x=x_orders, y=orders['recommended_order'], name="Replenishment Order (Qty)", marker_color='#2CA02C', opacity=0.9, text=orders['recommended_order'].astype(int), textposition='auto'), secondary_y=True)

    fig.update_layout(title=dict(text=f"<b>🛡️ Inventory Strategy & Replenishment Plan: {item}</b>", font=dict(size=20)), template="plotly_white", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.5)"), height=550)
    fig.update_yaxes(title_text="<b>Stock Level (Units)</b>", secondary_y=False, showgrid=True, gridcolor='#F0F0F0')
    max_order = plan['recommended_order'].max() if not plan['recommended_order'].empty else 10
    fig.update_yaxes(title_text="<b>Order Quantity</b>", secondary_y=True, showgrid=False, range=[0, max_order * 1.5])
    return fig

def plot_feature_importance(model, features, model_type):
    if model_type in ['Prophet', 'TF-C (Transformer)']: return None
        
    if model_type == 'LightGBM':
        importance = model.feature_importance(importance_type='gain')
    elif model_type == 'XGBoost':
        importance_map = model.get_score(importance_type='gain')
        importance = [importance_map.get(f, 0) for f in features]
    elif model_type == 'CatBoost':
        importance = model.get_feature_importance()
        
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=True).tail(15)
    fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='<b>Model Explainability: Top Drivers</b>', color='Importance', color_continuous_scale='Viridis')
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

    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_store = st.selectbox("Store Location", data['store_id'].unique())
        selected_item = st.selectbox("Product SKU", data['item_id'].unique())
        
        st.subheader("Model Hyperparameters")
        selected_model = st.radio("AI Architecture", ["LightGBM", "XGBoost", "CatBoost", "Prophet", "TF-C (Transformer)"], index=0)
        
        if selected_model == "TF-C (Transformer)":
            st.markdown("---")
            st.markdown("**TF-C Deep Learning Params**")
            epochs_pre = st.slider("Pre-training Epochs (Contrastive)", 1, 30, 5)
            epochs_fine = st.slider("Fine-Tuning Epochs (MSE)", 1, 30, 5)
        
        st.markdown("---")
        st.subheader("Inventory Simulation Parameters")
        lead_time = st.slider("Supplier Lead Time (Days)", 1, 14, 3)
        service_level = st.slider("Target Service Level (%)", 80, 99, 95)
        current_stock = st.number_input("Current Stock On-Hand", min_value=0, value=50)
        z_score = {99: 2.33, 95: 1.64, 90: 1.28, 85: 1.04, 80: 0.84}.get(service_level, 1.64)

    if st.button("🚀 Run AI Analysis", type="primary"):
        with st.status("Initializing AI Cortex...", expanded=True) as status:
            st.write("🔧 Preparing Data Grid...")
            grid = prepare_training_data(data, selected_item, selected_store)
            SPLIT_DAY = 1913
            
            # --- BRANCHING LOGIC ---
            if selected_model == "Prophet":
                st.write(f"🧠 Training {selected_model} (Bayesian Time Series)...")
                train_mask = grid['d_num'] <= SPLIT_DAY
                model = train_prophet_model(grid[train_mask])
                
                val_mask = (grid['d_num'] > SPLIT_DAY - 28) & (grid['d_num'] <= SPLIT_DAY)
                val_data = grid[val_mask][['date', 'sell_price']].copy()
                val_data.columns = ['ds', 'sell_price']
                val_preds = model.predict(val_data)['yhat'].values
                y_val = grid[val_mask]['sales'].values
                rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                
                st.write("🔮 Generating Batch Forecast (28 Days Horizon)...")
                forecast_grid, final_features = batch_predict_prophet(model, grid.copy(), 1914)

            elif selected_model == "TF-C (Transformer)":
                st.write(f"🧠 Initiating Deep Learning: {selected_model}")
                train_df = grid[grid['d_num'] <= SPLIT_DAY].copy()
                
                seq_len = 60
                if len(train_df) < seq_len * 2:
                    st.error("Insufficient historical data to train the Transformer sequence windows.")
                    status.update(label="Failed.", state="error")
                    return
                
                X_cont, X_cat, X_freq, y, scalers, cat_cards, cat_cols = prepare_tfc_tensors(train_df, seq_len=seq_len)
                
                log_placeholder = st.empty()
                pbar = st.progress(0, text="Initializing Weights...")
                
                model, head, val_mse = train_tfc_model(
                    X_cont, X_cat, X_freq, y, cat_cards, 
                    log_placeholder, pbar, 
                    epochs_pre=epochs_pre, epochs_fine=epochs_fine
                )
                
                rmse = np.sqrt(val_mse) * scalers['sales'].scale_[0] 
                
                st.write("🔮 Generating Autoregressive Sequence Forecast...")
                forecast_grid = recursive_predict_tfc(model, head, grid.copy(), scalers, cat_cols, 1914, seq_len=seq_len)
                final_features = ["Time Domain Tensors (w/ Positional Encoding)", "Frequency Domain FFT Tensors", "Cross-Domain Contrastive Embedded Space"]
                pbar.empty()

            else:
                st.write("🔧 Engineering temporal features (Lags, Rolling Windows)...")
                grid_feat = create_features(grid).dropna(subset=['lag_28', 'rolling_mean_28', 'sales'])
                exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'sales', 'd', 'date', 'wm_yr_wk', 'd_num']
                features = [c for c in grid_feat.columns if c not in exclude_cols]
                
                train_mask = grid_feat['d_num'] <= SPLIT_DAY - 28
                val_mask = (grid_feat['d_num'] > SPLIT_DAY - 28) & (grid_feat['d_num'] <= SPLIT_DAY)
                X_train, y_train = grid_feat[train_mask][features], grid_feat[train_mask]['sales']
                X_val, y_val = grid_feat[val_mask][features], grid_feat[val_mask]['sales']
                
                st.write(f"🧠 Training {selected_model} (Tweedie Loss Optimization)...")
                if selected_model == "LightGBM":
                    model = train_lightgbm(X_train, y_train, X_val, y_val)
                    val_preds = model.predict(X_val)
                elif selected_model == "XGBoost":
                    model = train_xgboost(X_train, y_train, X_val, y_val)
                    val_preds = model.predict(xgb.DMatrix(X_val, enable_categorical=True))
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
        next_order_day = next_order['date'].dt.strftime('%Y-%m-%d').iloc[0] if not next_order.empty else "None"
        
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
            cols = ['sales', 'safety_stock', 'projected_inventory', 'recommended_order']
            disp = plan[['date'] + cols].head(14).copy()
            disp['date'] = disp['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(disp.style.background_gradient(cmap='Reds', subset=['recommended_order']).format({c: "{:.1f}" for c in cols}), use_container_width=True)
            
        with tab3:
            if selected_model in ["Prophet", "TF-C (Transformer)"]:
                st.info(f"**{selected_model}** utilizes complex internal mechanisms (Bayesian curves / Neural Embeddings). Standard tree-based Feature Importance mapping is not natively applicable.")
                if selected_model == "TF-C (Transformer)":
                    st.markdown("**TF-C Architecture:** Relies on cross-referencing temporal autoregression (Time-Domain) with signal harmonics (Frequency-Domain FFT) using Contrastive Learning. Sequence structure mapped via Positional Encoding.")
            else:
                st.plotly_chart(plot_feature_importance(model, final_features, selected_model), use_container_width=True)
                st.info("Feature Importance shows which variables (Lags, Trends, Price) most influenced the AI's decision.")

if __name__ == "__main__":
    main()