import numpy as np
import pandas as pd

def generate_data(n=100):
    np.random.seed(42)
    
    # Terrain features
    slope = np.random.uniform(20, 60, n)   # slope angle in degrees
    aspect = np.random.uniform(0, 360, n)  # slope direction
    curvature = np.random.uniform(-1, 1, n)

    # Sensor & environment
    displacement = np.random.normal(0.5, 0.2, n)   # cm/day
    rainfall = np.random.uniform(0, 200, n)        # mm/24h
    temp = np.random.uniform(10, 40, n)

    # Risk label (synthetic rule)
    risk = ((slope > 40) & (rainfall > 100) & (displacement > 0.6)).astype(int)

    df = pd.DataFrame({
        "slope": slope,
        "aspect": aspect,
        "curvature": curvature,
        "displacement": displacement,
        "rainfall": rainfall,
        "temp": temp,
        "label": risk
    })
    df.to_csv("synthetic_data.csv", index=False)
    return df

if __name__ == "__main__":
    print(generate_data(20).head())
