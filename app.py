from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

# Load dataset and preprocess it
file_path = 'C:/Users/vuday/OneDrive/Desktop/Mini project/Project/electriccars.csv'  # Update with your actual file path
df = pd.read_csv(file_path)
features = ['Price(In lakhs)', 'Range(In km)', 'Battery Capacity(in kwh)', 'Seating Capacity', 'Max Power(bhp)']
df_original = df[:].copy()  # Store original data for use later
df = df.dropna()  # Handle missing values

# Calculate medians for each feature
medians = df_original[features].median()

# Normalize the features with a higher weight for Price
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Apply additional weight to Price
price_weight = 10
df['Weighted Price'] = df['Price(In lakhs)'] * price_weight

# Use the adjusted features for clustering
adjusted_features = ['Weighted Price'] + features[1:]
kmeans = KMeans(n_clusters=5, random_state=0)
df['Cluster'] = kmeans.fit_predict(df[adjusted_features])

@app.route('/')
def index():
    return render_template('explore.html')

@app.route('/explore', methods=['GET', 'POST'])
def explore():
    if request.method == 'POST':
        try:
            input_features = [
                request.form['Price'],
                request.form['Range'],
                request.form['Battery Capacity'],
                request.form['Seating Capacity'],
                request.form['Max Power']
            ]
            
            # Replace missing values with medians
            input_features = [
                float(input_features[0]) if input_features[0] else medians['Price(In lakhs)'],
                int(input_features[1]) if input_features[1] else medians['Range(In km)'],
                int(input_features[2]) if input_features[2] else medians['Battery Capacity(in kwh)'],
                int(input_features[3]) if input_features[3] else medians['Seating Capacity'],
                int(input_features[4]) if input_features[4] else medians['Max Power(bhp)']
            ]

            car_details = find_nearest_cars(input_features)
            return render_template('results.html', cars=car_details)
        except Exception as e:
            print(f"Error processing input: {e}")
            return redirect(url_for('explore'))
    return render_template('explore.html')

def find_nearest_cars(input_features, num_results=5):
    input_features_df = pd.DataFrame([input_features], columns=features)
    input_features_normalized = scaler.transform(input_features_df)[0]
    
    # Apply the same weight adjustment to the input price
    input_features_normalized[0] *= price_weight
    
    distances = [np.sqrt(np.sum((input_features_normalized - centroid) ** 2)) for centroid in kmeans.cluster_centers_]
    nearest_cluster = np.argmin(distances)
    
    nearest_cars = df[df['Cluster'] == nearest_cluster].copy()
    nearest_cars['Distance'] = nearest_cars.apply(
        lambda row: np.sqrt(np.sum((row[adjusted_features] - np.append(input_features_normalized[0], input_features_normalized[1:])) ** 2)),
        axis=1
    )
    nearest_cars = nearest_cars.nsmallest(num_results, 'Distance')
    
    # Map normalized values back to original values
    original_nearest_cars = df_original.iloc[nearest_cars.index].copy()
    original_nearest_cars['Distance'] = nearest_cars['Distance'].values  # Keep distances for sorting
    
    car_details = original_nearest_cars.to_dict(orient='records')
    
    return car_details

if __name__ == '__main__':
    app.run(debug=True)
