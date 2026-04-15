import pandas as pd

# Load both CSVs
print("Loading games.csv ...")
games = pd.read_csv("games.csv")

print("Loading games-list.csv ...")
games_list = pd.read_csv("games-list.csv")

# Only keep the columns from games-list that games does NOT have
extra_cols = ["unreleased", "firstReleaseDate", "earlyAccess",
              "copiesSold", "reviewScore", "publisherClass", "steamUrl"]

games_list_slim = games_list[["steamId"] + extra_cols]

# LEFT JOIN: keep every row in games, match on AppID == steamId
merged = games.merge(games_list_slim, left_on="AppID", right_on="steamId", how="left")

# Drop the redundant steamId column (same as AppID)
merged.drop(columns=["steamId"], inplace=True)

# Move copiesSold to the 3rd column (index 2)
cols = list(merged.columns)
cols.remove("copiesSold")
cols.insert(2, "copiesSold")
merged = merged[cols]

# Save result
output = "games_merged.csv"
merged.to_csv(output, index=False)

print(f"\nDone! {len(merged)} rows saved → {output}")
print(f"  games.csv rows:      {len(games)}")
print(f"  games-list.csv rows: {len(games_list)}")
print(f"  Matched rows:        {merged['copiesSold'].notna().sum()}")
print(f"  Unmatched rows:      {merged['copiesSold'].isna().sum()}")
