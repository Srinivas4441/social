from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import io
import base64

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("data/test1.txt")

def analyze_platform(platform):
    # Convert first letter to uppercase if needed
    platform = platform.capitalize()

    # Filter the dataframe for the selected platform
    platform_df = df[df['Platform'] == platform]

    if platform_df.empty:
        return f"No data available for platform: {platform}", None

    # Define features (X) and target (y)
    X = platform_df[['Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']]
    y = platform_df['Daily_Usage_Time (minutes)']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plotting the actual vs predicted values
    plt.figure(figsize=(10, 6))
    sns.barplot(x=y_test.index, y=y_test, color='blue', alpha=0.6, label='Actual')
    sns.barplot(x=y_test.index, y=y_pred, color='red', alpha=0.6, label='Predicted')
    plt.xlabel(f'{platform}')
    plt.ylabel('Daily Usage Time (minutes)')
    plt.title(f'Actual vs Predicted Daily Usage Time for {platform}')
    plt.legend()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')

    result_summary = f"Results for {platform}:<br>" \
                     f"Mean squared error: {mse:.2f}<br>" \
                     f"R^2 score: {r2:.2f}"

    return result_summary, plot_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    platform = request.form['platform']
    result_summary, plot_data = analyze_platform(platform)
    return render_template('results.html', result_summary=result_summary, plot_data=plot_data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
