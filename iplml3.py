import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
import os


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


deliveries_path = r"D:\ipl-score-predictor\ipl_data3\deliveries.csv"
matches_path = r"D:\ipl-score-predictor\ipl_data3\matches.csv"


try:
    deliveries = pd.read_csv(deliveries_path, encoding='utf-8')
    matches = pd.read_csv(matches_path, encoding='utf-8')
except UnicodeDecodeError:
    deliveries = pd.read_csv(deliveries_path, encoding='ISO-8859-1')
    matches = pd.read_csv(matches_path, encoding='ISO-8859-1')


df = deliveries.merge(matches, left_on="match_id", right_on="id")


df = df[df["inning"] == 1]


df["dismissed"] = df["player_dismissed"].notnull().astype(int)


grouped = df.groupby(["match_id", "over"]).agg({
    "batting_team": "first",
    "bowling_team": "first",
    "total_runs": "sum",
    "dismissed": "sum"
}).reset_index()


grouped["cumulative_runs"] = grouped.groupby("match_id")["total_runs"].cumsum()
grouped["cumulative_wickets"] = grouped.groupby("match_id")["dismissed"].cumsum()
grouped["run_rate"] = grouped["cumulative_runs"] / grouped["over"]


data_after_over = grouped[grouped["over"] == 6]


final_scores = grouped.groupby("match_id")["cumulative_runs"].max().reset_index()
final_scores.rename(columns={"cumulative_runs": "total_score"}, inplace=True)


df_model = pd.merge(data_after_over, final_scores, on="match_id")


teams = sorted(df_model["batting_team"].unique())
team_to_code = {team: idx for idx, team in enumerate(teams)}
code_to_team = {idx: team for team, idx in team_to_code.items()}

df_model["batting_team_code"] = df_model["batting_team"].map(team_to_code)
df_model["bowling_team_code"] = df_model["bowling_team"].map(team_to_code)


features = ["batting_team_code", "bowling_team_code", "over", "cumulative_runs", "cumulative_wickets", "run_rate"]
X = df_model[features]
y = df_model["total_score"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"\n✅ MAE: {mae:.2f}")
print(f"✅ RMSE: {rmse:.2f}")


print("\n🎯 IPL Score Prediction (After 6 Overs) 🎯")
print("\nAvailable Teams:")
for team in team_to_code:
    print(f"- {team}")


batting_team_input = input("\nEnter Batting Team: ").strip()
bowling_team_input = input("Enter Bowling Team: ").strip()


if batting_team_input not in team_to_code or bowling_team_input not in team_to_code:
    print("❌ Invalid team name. Please choose from the listed teams.")
    exit()

runs_input = int(input("Enter Runs Scored after 6 overs: "))
wickets_input = int(input("Enter Wickets Fallen after 6 overs: "))
run_rate_input = float(input("Enter Current Run Rate: "))


input_data = [[
    team_to_code[batting_team_input],
    team_to_code[bowling_team_input],
    6,
    runs_input,
    wickets_input,
    run_rate_input
]]

predicted_score = model.predict(input_data)[0]
print(f"\n📢 Predicted Final Score: {predicted_score:.2f}")


model.save_model("ipl_model.json")


