import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Bitcoin LSTM Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BitcoinLSTMPredictor:
    def __init__(self, db_path="bitcoin_data.db"):
        self.db_path = db_path
        self.model = None
        self.scaler = None
        self.sequence_length = 60
        
    def create_database(self):
        """Create SQLite database and table for Bitcoin data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bitcoin_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume_btc REAL,
                volume_usd REAL,
                weighted_price REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_data_to_db(self, csv_file):
        """Load CSV data into SQLite database"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Clean column names
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
            
            # Handle different possible column names
            column_mapping = {
                'volume_btc': ['volume_btc', 'volume_bitcoin', 'volume_btc_'],
                'volume_usd': ['volume_usd', 'volume_currency', 'volume_usd_'],
                'weighted_price': ['weighted_price', 'weighted_price_usd']
            }
            
            for standard_name, possible_names in column_mapping.items():
                for possible_name in possible_names:
                    if possible_name in df.columns:
                        df = df.rename(columns={possible_name: standard_name})
                        break
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    st.error(f"Required column '{col}' not found in the dataset")
                    return False
            
            # Handle missing optional columns
            if 'volume_btc' not in df.columns:
                df['volume_btc'] = 0
            if 'volume_usd' not in df.columns:
                df['volume_usd'] = 0
            if 'weighted_price' not in df.columns:
                df['weighted_price'] = df['close']
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Convert timestamp to integer if needed
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
            
            # Connect to database and insert data
            conn = sqlite3.connect(self.db_path)
            
            # Clear existing data
            conn.execute("DELETE FROM bitcoin_data")
            
            # Insert new data
            df.to_sql('bitcoin_data', conn, if_exists='append', index=False, 
                     method='multi', chunksize=10000)
            
            conn.commit()
            conn.close()
            
            st.success(f"Successfully loaded {len(df)} records into database")
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def get_data_from_db(self, limit=None):
        """Retrieve data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, open, high, low, close, volume_btc, volume_usd, weighted_price
            FROM bitcoin_data 
            ORDER BY timestamp
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def prepare_data(self, df, target_column='close'):
        """Prepare data for LSTM training"""
        # Use only the target column for prediction
        data = df[target_column].values.reshape(-1, 1)
        
        # Scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaled_data
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the LSTM model"""
        self.model = self.build_model((X_train.shape[1], 1))
        
        # Create callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def save_model(self, model_path="bitcoin_lstm_model.h5", scaler_path="scaler.pkl"):
        """Save trained model and scaler"""
        if self.model:
            self.model.save(model_path)
        if self.scaler:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def load_model(self, model_path="bitcoin_lstm_model.h5", scaler_path="scaler.pkl"):
        """Load trained model and scaler"""
        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            return True
        except:
            return False

def main():
    st.title("₿ Bitcoin LSTM Price Predictor")
    st.markdown("---")
    
    # Initialize predictor
    predictor = BitcoinLSTMPredictor()
    predictor.create_database()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Data Upload", "Model Training", "Predictions", "Analysis"])
    
    if page == "Data Upload":
        st.header("📊 Data Upload and Management")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Bitcoin CSV file", type=['csv'])
        
        if uploaded_file is not None:
            if st.button("Load Data to Database"):
                with st.spinner("Loading data to database..."):
                    success = predictor.load_data_to_db(uploaded_file)
                
                if success:
                    # Show data preview
                    df = predictor.get_data_from_db(limit=1000)
                    st.subheader("Data Preview (First 1000 records)")
                    st.dataframe(df.head())
                    
                    # Show basic statistics
                    st.subheader("Basic Statistics")
                    st.write(df.describe())
        
        # Show current database status
        try:
            conn = sqlite3.connect(predictor.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM bitcoin_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            st.info(f"Current database contains {count:,} records")
            
            if count > 0:
                df_sample = predictor.get_data_from_db(limit=5)
                st.subheader("Latest 5 records in database:")
                st.dataframe(df_sample.tail())
                
        except:
            st.warning("No data in database yet. Please upload a CSV file.")
    
    elif page == "Model Training":
        st.header("🤖 Model Training")
        
        try:
            conn = sqlite3.connect(predictor.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM bitcoin_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count == 0:
                st.warning("No data available. Please upload data first.")
                return
                
        except:
            st.error("Database not accessible. Please upload data first.")
            return
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 10, 200, 50)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        
        with col2:
            test_size = st.slider("Test Size (%)", 10, 40, 20)
            sequence_length = st.slider("Sequence Length", 30, 120, 60)
        
        predictor.sequence_length = sequence_length
        
        if st.button("Start Training"):
            with st.spinner("Loading data and training model..."):
                # Load data
                df = predictor.get_data_from_db()
                st.info(f"Loaded {len(df)} records for training")
                
                # Prepare data
                X, y, scaled_data = predictor.prepare_data(df)
                
                # Split data
                split_index = int(len(X) * (1 - test_size/100))
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]
                
                st.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
                
                # Train model
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.4f} - Val Loss: {logs["val_loss"]:.4f}')
                
                # Add custom callback
                callbacks = [StreamlitCallback()]
                
                history = predictor.train_model(X_train, y_train, X_test, y_test, 
                                              epochs=epochs, batch_size=batch_size)
                
                # Save model
                predictor.save_model()
                
                # Show training results
                st.success("Model training completed!")
                
                # Plot training history
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Mean Absolute Error'))
                
                fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'), row=1, col=1)
                fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'), row=1, col=1)
                fig.add_trace(go.Scatter(y=history.history['mae'], name='Train MAE'), row=1, col=2)
                fig.add_trace(go.Scatter(y=history.history['val_mae'], name='Val MAE'), row=1, col=2)
                
                fig.update_layout(title="Training History", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Evaluate model
                train_predictions = predictor.predict(X_train)
                test_predictions = predictor.predict(X_test)
                
                train_actual = predictor.scaler.inverse_transform(y_train.reshape(-1, 1))
                test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
                
                train_rmse = np.sqrt(mean_squared_error(train_actual, train_predictions))
                test_rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
                
                train_mae = mean_absolute_error(train_actual, train_predictions)
                test_mae = mean_absolute_error(test_actual, test_predictions)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Train RMSE", f"${train_rmse:.2f}")
                col2.metric("Test RMSE", f"${test_rmse:.2f}")
                col3.metric("Train MAE", f"${train_mae:.2f}")
                col4.metric("Test MAE", f"${test_mae:.2f}")
    
    elif page == "Predictions":
        st.header("🔮 Price Predictions")
        
        # Load model
        if not predictor.load_model():
            st.warning("No trained model found. Please train a model first.")
            return
        
        # Load recent data
        df = predictor.get_data_from_db()
        if len(df) == 0:
            st.warning("No data available for predictions.")
            return
        
        # Prediction parameters
        col1, col2 = st.columns(2)
        with col1:
            days_to_predict = st.slider("Days to predict", 1, 30, 7)
        with col2:
            show_last_days = st.slider("Show last N days", 30, 365, 90)
        
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                # Prepare data for prediction
                X, y, scaled_data = predictor.prepare_data(df)
                
                # Get recent data for prediction
                recent_data = scaled_data[-predictor.sequence_length:]
                
                # Generate future predictions
                future_predictions = []
                current_sequence = recent_data.copy()
                
                for _ in range(days_to_predict):
                    # Reshape for prediction
                    input_seq = current_sequence.reshape(1, predictor.sequence_length, 1)
                    
                    # Predict next value
                    next_pred = predictor.model.predict(input_seq, verbose=0)
                    future_predictions.append(next_pred[0, 0])
                    
                    # Update sequence
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[-1] = next_pred[0, 0]
                
                # Convert predictions back to original scale
                future_predictions = np.array(future_predictions).reshape(-1, 1)
                future_predictions = predictor.scaler.inverse_transform(future_predictions)
                
                # Create future dates
                last_date = pd.to_datetime(df['timestamp'].iloc[-1], unit='s')
                future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
                
                # Plot predictions
                fig = go.Figure()
                
                # Historical data
                recent_df = df.tail(show_last_days)
                fig.add_trace(go.Scatter(
                    x=recent_df['date'],
                    y=recent_df['close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                
                # Future predictions
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions.flatten(),
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f'Bitcoin Price Prediction - Next {days_to_predict} Days',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction table
                st.subheader("Prediction Summary")
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions.flatten()
                })
                pred_df['Predicted Price'] = pred_df['Predicted Price'].apply(lambda x: f"${x:.2f}")
                st.dataframe(pred_df)
                
                # Show current price and prediction summary
                current_price = df['close'].iloc[-1]
                avg_predicted_price = future_predictions.mean()
                price_change = avg_predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("Avg Predicted Price", f"${avg_predicted_price:.2f}")
                col3.metric("Expected Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    
    elif page == "Analysis":
        st.header("📈 Data Analysis")
        
        # Load data
        df = predictor.get_data_from_db()
        if len(df) == 0:
            st.warning("No data available for analysis.")
            return
        
        # Basic statistics
        st.subheader("Market Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
        col3.metric("All-Time High", f"${df['high'].max():.2f}")
        col4.metric("All-Time Low", f"${df['low'].min():.2f}")
        
        # Price chart
        st.subheader("Price History")
        
        # Time range selector
        time_range = st.selectbox("Select Time Range", 
                                 ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "All Time"])
        
        if time_range == "Last 30 Days":
            plot_df = df.tail(30*24*60)  # Assuming minute data
        elif time_range == "Last 90 Days":
            plot_df = df.tail(90*24*60)
        elif time_range == "Last 6 Months":
            plot_df = df.tail(180*24*60)
        elif time_range == "Last Year":
            plot_df = df.tail(365*24*60)
        else:
            plot_df = df
        
        # Candlestick chart
        fig = go.Figure(data=go.Candlestick(
            x=plot_df['date'],
            open=plot_df['open'],
            high=plot_df['high'],
            low=plot_df['low'],
            close=plot_df['close']
        ))
        
        fig.update_layout(
            title='Bitcoin Price Chart',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume analysis
        if 'volume_usd' in df.columns:
            st.subheader("Volume Analysis")
            
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Scatter(
                x=plot_df['date'],
                y=plot_df['volume_usd'],
                mode='lines',
                name='Volume (USD)',
                fill='tonexty'
            ))
            
            fig_volume.update_layout(
                title='Trading Volume',
                xaxis_title='Date',
                yaxis_title='Volume (USD)',
                height=400
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # Price statistics
        st.subheader("Price Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Standard Deviation', 'Volatility (%)'],
            'Close Price': [
                f"${df['close'].mean():.2f}",
                f"${df['close'].median():.2f}",
                f"${df['close'].std():.2f}",
                f"{(df['close'].std() / df['close'].mean() * 100):.2f}%"
            ]
        })
        st.dataframe(stats_df)

if __name__ == "__main__":
    main()
