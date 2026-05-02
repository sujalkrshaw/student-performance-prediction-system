import pandas as pd
import numpy as np

np.random.seed(42)

# Create synthetic data
data = pd.DataFrame({
    "study_hours": np.random.randint(1, 10, 300),
    "attendance": np.random.randint(40, 100, 300),
    "assignments": np.random.randint(30, 100, 300),
    "quiz_score": np.random.randint(20, 100, 300)
})

# 🔥 Add realistic behavior + randomness
data["final_score"] = (
    data["study_hours"] * 4 +
    data["attendance"] * 0.2 +
    data["assignments"] * 0.3 +
    data["quiz_score"] * 0.5 +
    np.random.normal(0, 10, len(data))   # 🔥 noise added
)

# 🔥 Add interaction effect (important for realism)
data["final_score"] += (data["study_hours"] * data["attendance"]) * 0.01

# Normalize score (optional but cleaner)
data["final_score"] = data["final_score"].clip(0, 100)

# Convert to Pass/Fail with better threshold
data["result"] = data["final_score"].apply(lambda x: 1 if x > 55 else 0)

# Save dataset
data.to_csv("data/students.csv", index=False)

print("✅ Improved dataset created successfully!")