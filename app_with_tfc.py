import streamlit as st
import pandas as pd
import numpy as np
import math
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
import logging

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 0. High-Performance Configuration
# ==========================================
st.set_page_config(
    page_title="Supply Chain AI Cortex | Enterprise Matrix",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)
warnings.filterwarnings('ignore')

logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.markdown("""
<style>
    .stMetric {
        background-color: #0E1117;
        border: 1px solid #30333F;
        padding: 15px;
        border-radius: 5px;
    }
    .stProgress .st-bo { background-color: #00AA00; }
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
# 1. Global Data Pipeline & Temporal Engineering
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
    df = df.copy().sort_values(by=['item_id', 'd_num'])
    
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    df['dayofmonth'] = df['date'].dt.day 

    df['lag_1'] = df.groupby('item_id')['sales'].shift(1)
    df['lag_7'] = df.groupby('item_id')['sales'].shift(7)
    df['lag_14'] = df.groupby('item_id')['sales'].shift(14)
    df['lag_28'] = df.groupby('item_id')['sales'].shift(28)
    
    df['shifted_sales'] = df['lag_1']
    df['rolling_mean_7'] = df.groupby('item_id')['shifted_sales'].transform(lambda x: x.rolling(7).mean())
    df['rolling_std_7'] = df.groupby('item_id')['shifted_sales'].transform(lambda x: x.rolling(7).std())
    df['rolling_mean_28'] = df.groupby('item_id')['shifted_sales'].transform(lambda x: x.rolling(28).mean())
    
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
    
    # PERFECTION LAYER: We intentionally leave these as raw integers here. 
    # We will cast them mathematically right before training to preserve schema indexing.
    return data, le_item

# ==========================================
# 2. PyTorch TF-C Architecture (Globalized)
# ==========================================
class PositionalEncoding(nn.Module):
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
            nn.Linear(d_model * seq_len, 128), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 64)
        )

        self.freq_input_proj = nn.Linear(1, d_model)
        encoder_layer_f = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*2, 
            batch_first=True, dropout=0.2, norm_first=True
        )
        self.freq_encoder = nn.TransformerEncoder(encoder_layer_f, num_layers=2)
        self.projector_f = nn.Sequential(
            nn.Linear(d_model * freq_len, 128), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 64)
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
            nn.Linear(time_dim + freq_dim, 128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, 1)
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
        return self.criterion(similarity_matrix, labels) / (2 * batch_size)

# --- Global TF-C Engineering ---
def prepare_global_tfc_tensors(df, seq_len=60):
    sales_scaler = StandardScaler()
    price_scaler = StandardScaler()
    
    work_df = df.copy().sort_values(by=['item_id', 'd_num'])
    work_df['sales'] = work_df['sales'].fillna(0)
    work_df['sell_price'] = work_df.groupby('item_id')['sell_price'].fillna(method='ffill').fillna(0)
    
    work_df['sales_scaled'] = sales_scaler.fit_transform(work_df[['sales']])
    work_df['price_scaled'] = price_scaler.fit_transform(work_df[['sell_price']])
    
    cat_cols = ['item_id_encoded', 'event_name_1', 'dayofweek', 'is_weekend', 'month', 'dayofmonth']
    cat_cards = [work_df[col].max() + 1 for col in cat_cols]
    
    X_cont, X_cat, X_freq, y = [], [], [], []
    
    for item_id, group in work_df.groupby('item_id'):
        data_cont = group[['sales_scaled', 'price_scaled']].values
        data_cat = group[cat_cols].values.astype(int)
        
        for i in range(len(group) - seq_len):
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
    return X_cont_t, X_cat_t, X_freq_t, y_t, scalers, cat_cards, cat_cols

def train_global_tfc_model(X_cont, X_cat, X_freq, y, cat_cards, log_placeholder, pbar, epochs_pre=10, epochs_fine=15):
    dataset = TensorDataset(X_cont, X_cat, X_freq, y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    model = AdvancedTFC(seq_len=X_cont.shape[1], freq_len=X_freq.shape[1], 
                        num_cont_features=X_cont.shape[2], cat_cardinalities=cat_cards, d_model=64).to(DEVICE)
    head = ForecasterHead(64 * X_cont.shape[1], 64 * X_freq.shape[1]).to(DEVICE)
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-5)
    
    log_text = "<b>[SYSTEM] Initializing Global TF-C Tensor Training...</b><br>"
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
        pbar.progress(current_epoch / total_epochs, text="Phase 1: Global Contrastive Pre-Training...")
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
            loss = criterion(head(h_t, h_f), yt)
            
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
        pbar.progress(current_epoch / total_epochs, text="Phase 2: Global Supervised Fine-Tuning...")
        log_text = f"[Phase 2] Epoch {epoch+1}/{epochs_fine} | Train MSE: {total_mse/len(train_loader):.4f} | Val MSE: {final_val_mse:.4f}<br>" + log_text
        log_placeholder.markdown(f"<div class='log-terminal'>{log_text}</div>", unsafe_allow_html=True)

    return model, head, final_val_mse

def recursive_predict_global_tfc(model, head, grid, scalers, cat_cols, selected_items, start_day, horizon, seq_len=60):
    model.eval()
    head.eval()
    sim_grid = grid.copy().sort_values(by=['item_id', 'd_num'])
    progress_bar = st.progress(0, text="Running Vectorized TF-C Simulation...")
    
    for i in range(horizon):
        target_day = start_day + i
        batch_xc, batch_xcat, batch_xf = [], [], []
        
        for item in selected_items:
            history_window = sim_grid[(sim_grid['item_id'] == item) & (sim_grid['d_num'] < target_day)].tail(seq_len)
            
            s_data = scalers['sales'].transform(history_window[['sales']].fillna(0)).reshape(-1, 1)
            p_data = scalers['price'].transform(history_window[['sell_price']].fillna(0)).reshape(-1, 1)
            batch_xc.append(np.hstack([s_data, p_data]))
            
            batch_xcat.append(history_window[cat_cols].values.astype(int))
            
            fft = np.abs(np.fft.rfft(s_data.flatten()))
            batch_xf.append((fft - np.mean(fft)) / (np.std(fft) + 1e-6))
            
        xc_t = torch.FloatTensor(np.array(batch_xc)).to(DEVICE)
        xcat_t = torch.LongTensor(np.array(batch_xcat)).to(DEVICE)
        xf_t = torch.FloatTensor(np.array(batch_xf)).unsqueeze(-1).to(DEVICE)
        
        with torch.no_grad():
            ht, _, hf, _ = model(xc_t, xcat_t, xf_t)
            preds_scaled = head(ht, hf).cpu().numpy()
            
        preds_real = scalers['sales'].inverse_transform(preds_scaled).flatten()
        preds_real = np.maximum(0, preds_real)
        
        mask = (sim_grid['d_num'] == target_day) & (sim_grid['item_id'].isin(selected_items))
        sim_grid.loc[mask, 'sales'] = preds_real
        progress_bar.progress((i + 1) / horizon, text=f"Simulating Day {target_day} across Deep Tensor...")
        
    progress_bar.empty()
    return sim_grid

# ==========================================
# 3. Global Tree Architectures
# ==========================================
def train_lightgbm_global(X_train, y_train, X_val, y_val, lr, iterations):
    params = {'objective': 'tweedie', 'tweedie_variance_power': 1.1, 'metric': 'rmse', 'learning_rate': lr, 'num_leaves': 128, 'max_depth': 8, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.5, 'lambda_l2': 0.5, 'n_jobs': -1, 'verbosity': -1}
    cat_features = ['item_id_encoded', 'event_name_1', 'dayofweek', 'is_weekend', 'month', 'dayofmonth']
    valid_cats = [c for c in cat_features if c in X_train.columns]
    return lgb.train(params, lgb.Dataset(X_train, label=y_train, categorical_feature=valid_cats), num_boost_round=iterations, valid_sets=[lgb.Dataset(X_train, label=y_train, categorical_feature=valid_cats), lgb.Dataset(X_val, label=y_val, reference=lgb.Dataset(X_train, label=y_train, categorical_feature=valid_cats), categorical_feature=valid_cats)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

def train_xgboost_global(X_train, y_train, X_val, y_val, lr, iterations):
    y_train, y_val = y_train.astype(float), y_val.astype(float)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    params = {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.1, 'eval_metric': 'rmse', 'eta': lr, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8, 'alpha': 0.5, 'lambda': 0.5, 'nthread': -1, 'verbosity': 0}
    return xgb.train(params, dtrain, num_boost_round=iterations, evals=[(dtrain, 'train'), (dval, 'eval')], early_stopping_rounds=100, verbose_eval=False)

def recursive_predict_global_trees(model, df, start_day, horizon, model_type, cat_dtypes):
    """
    CRITICAL FIX: Enforces frozen categorical boundaries onto validation slices 
    to prevent Pandas fragmentation errors.
    """
    exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'sales', 'd', 'date', 'wm_yr_wk', 'd_num']
    pbar = st.progress(0, text="Running Vectorized Global Simulation...")
    
    for i, day in enumerate(range(start_day, start_day + horizon)):
        df = create_features_global(df)
        features = [c for c in df.columns if c not in exclude_cols]
        mask = df['d_num'] == day
        X_test = df[mask][features].copy()
        
        if X_test.empty: break
        
        # Enforce exact mathematical schema from the training environment
        for col, dtype in cat_dtypes.items():
            if col in X_test.columns:
                X_test[col] = X_test[col].astype(dtype)
        
        if model_type == 'LightGBM': 
            preds = model.predict(X_test)
        elif model_type == 'XGBoost': 
            preds = model.predict(xgb.DMatrix(X_test, enable_categorical=True))
        elif model_type == 'CatBoost': 
            preds = model.predict(X_test)
            
        df.loc[mask, 'sales'] = np.maximum(0, preds) 
        pbar.progress((i + 1) / horizon, text=f"Simulating Day {day} across Global Tensor...")
        
    pbar.empty()
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
# 4. Interactive Visualization
# ==========================================
def plot_portfolio_comparison(portfolio_details):
    fig = go.Figure()
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
    for idx, (item, details) in enumerate(portfolio_details.items()):
        plan = details['plan']
        x_axis = plan['date'] if 'date' in plan.columns else plan['d_num']
        fig.add_trace(go.Scatter(x=x_axis, y=plan['projected_inventory'], name=f"{item}", mode='lines', line=dict(width=2, color=colors[idx % len(colors)])))
    fig.update_layout(title=dict(text="<b>🌐 Global Portfolio Convergence</b>", font=dict(size=18)), template="plotly_white", hovermode="x unified", height=450, yaxis_title="Units on Hand", xaxis_title="Timeline")
    return fig

def plot_interactive_lifecycle(history, forecast, ground_truth, item, rmse):
    history_view = history[history['d_num'] > (history['d_num'].max() - 120)]
    reality = pd.concat([history_view, ground_truth]).dropna(subset=['sales']).sort_values(by='d_num')
    fig = go.Figure()
    x_real = reality['date'] if 'date' in reality.columns else reality['d_num']
    fig.add_trace(go.Scatter(x=x_real, y=reality['sales'], mode='lines', name='Actual Sales (Reality)', line=dict(color='gray', width=1.5), opacity=0.7))
    x_fore = forecast['date'] if 'date' in forecast.columns else forecast['d_num']
    fig.add_trace(go.Scatter(x=x_fore, y=forecast['sales'], mode='lines+markers', name='AI Forecast', line=dict(color='#00CC96', width=3), marker=dict(size=5)))
    upper_bound = forecast['sales'] + (1.96 * rmse)
    lower_bound = forecast['sales'].apply(lambda x: max(0, x - (1.96 * rmse)))
    fig.add_trace(go.Scatter(x=pd.concat([x_fore, x_fore[::-1]]), y=pd.concat([upper_bound, lower_bound[::-1]]), fill='toself', fillcolor='rgba(0, 204, 150, 0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='95% Confidence Interval'))
    fig.update_layout(title=f"<b>Deep Learning Trajectory vs Reality: {item}</b>", xaxis_title="Date", yaxis_title="Sales Volume", template="plotly_white", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=450)
    return fig

def plot_interactive_action_plan(plan, item):
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

# ==========================================
# 5. Main Application Logic
# ==========================================
def main():
    st.title("🧠 Supply Chain AI Cortex | Global Matrix Edition")
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
            selected_model = st.radio("Architecture", ["LightGBM", "XGBoost", "CatBoost", "TF-C (Transformer)"], index=0)
            
            if selected_model == "TF-C (Transformer)":
                st.markdown("---")
                st.markdown("**TF-C Deep Learning Params**")
                epochs_pre = st.slider("Pre-training Epochs (Contrastive)", 1, 30, 5)
                epochs_fine = st.slider("Fine-Tuning Epochs (MSE)", 1, 30, 5)
            else:
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
            grid, le_item = prepare_global_training_data(data, selected_items, selected_store)
            grid = extend_grid_global(grid, forecast_horizon, SPLIT_DAY + 1)
            grid = create_features_global(grid)
            
            # --- TF-C Transformer Execution Path ---
            if selected_model == "TF-C (Transformer)":
                st.write(f"🧠 Initiating Deep Learning Tensor Protocol: {selected_model}")
                train_df = grid[grid['d_num'] <= SPLIT_DAY].copy()
                seq_len = 60
                
                X_cont, X_cat, X_freq, y, scalers, cat_cards, cat_cols = prepare_global_tfc_tensors(train_df, seq_len=seq_len)
                
                log_placeholder = st.empty()
                pbar = st.progress(0, text="Initializing Weights...")
                
                model, head, val_mse = train_global_tfc_model(
                    X_cont, X_cat, X_freq, y, cat_cards, log_placeholder, pbar, epochs_pre, epochs_fine
                )
                
                base_rmse = np.sqrt(val_mse) * scalers['sales'].scale_[0]
                for item in selected_items: master_metrics['item_rmses'][item] = base_rmse 
                
                st.write("4. 🔮 Executing Vectorized Future Simulation...")
                forecast_grid = recursive_predict_global_tfc(model, head, grid.copy(), scalers, cat_cols, selected_items, SPLIT_DAY + 1, forecast_horizon, seq_len)
                pbar.empty()

            # --- Gradient Boosting Execution Path ---
            else:
                st.write("2. 🧮 Engineering Tree Features & Freezing Mathematical Dtypes...")
                grid_feat_train = grid.dropna(subset=['lag_28', 'rolling_mean_28', 'sales']).copy()
                exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'sales', 'd', 'date', 'wm_yr_wk', 'd_num']
                features = [c for c in grid_feat_train.columns if c not in exclude_cols]
                
                # --- The Protocol of Mizan: Enforcing Absolute Global Categorical Dtypes ---
                cat_features = ['item_id_encoded', 'event_name_1', 'dayofweek', 'is_weekend', 'month', 'dayofmonth']
                global_cat_dtypes = {}
                
                for c in cat_features:
                    if c in grid_feat_train.columns:
                        grid_feat_train[c] = grid_feat_train[c].astype('category')
                        global_cat_dtypes[c] = grid_feat_train[c].dtype # Freeze the schema!

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
                    valid_cats = [c for c in cat_features if c in features]
                    cat_indices = [features.index(c) for c in valid_cats]
                    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
                    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
                    model = CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=6, loss_function='Tweedie:variance_power=1.1', verbose=0, l2_leaf_reg=5)
                    model.fit(train_pool, eval_set=val_pool)
                    val_preds = model.predict(X_val)

                X_val_df = grid_feat_train[val_mask].copy()
                X_val_df['val_preds'] = val_preds
                for item in selected_items:
                    item_val = X_val_df[X_val_df['item_id'] == item]
                    rmse = np.sqrt(mean_squared_error(item_val['sales'], item_val['val_preds'])) if not item_val.empty else 0.5
                    master_metrics['item_rmses'][item] = rmse

                st.write("4. 🔮 Executing Vectorized Future Simulation...")
                forecast_grid, _ = recursive_predict_global_trees(model, grid.copy(), SPLIT_DAY + 1, forecast_horizon, selected_model, global_cat_dtypes)
            
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
            
            status.update(label=f"✅ Unified AI Processed {len(selected_items)} Nodes Matrix.", state="complete", expanded=False)

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
                st.plotly_chart(plot_interactive_lifecycle(details['history_grid'], details['plan'], details['ground_truth'], inspect_item, details['rmse']), use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(plot_interactive_action_plan(details['plan'], inspect_item), use_container_width=True)
        with tab2:
            st.markdown("#### 💾 Master Replenishment Plan")
            export_df = st.session_state['master_df'][['date', 'item_id', 'sales', 'safety_stock', 'projected_inventory', 'recommended_order']].copy()
            export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
            actionable_orders = export_df[export_df['recommended_order'] > 0].sort_values(by=['date', 'item_id'])
            
            st.download_button("📥 Download Global ERP Matrix (CSV)", data=export_df.to_csv(index=False).encode('utf-8'), file_name='global_portfolio_plan.csv', mime='text/csv', type="primary")
            st.dataframe(actionable_orders.style.background_gradient(cmap='Greens', subset=['recommended_order']).format(precision=1), use_container_width=True)

if __name__ == "__main__":
    main()